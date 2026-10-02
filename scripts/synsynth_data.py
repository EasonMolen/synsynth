"""
四个评估方向所用的合成与自适应数据集。

本模块：
  1. 尝试通过 Hugging Face `datasets` 加载 DocRED、TACRED、HotpotQA 等真实基准数据。
  2. 如果无法获取，则用 Gemma-4 生成有代表性的合成样本。
  3. 将所有数据缓存在 data/，方便复现实验。
"""
from __future__ import annotations

import json
import os
import random
from typing import Any

from synsynth_config import (
    DATA_DIR, NUM_EVAL_SAMPLES, RANDOM_SEED, safe_path, logger,
    NUM_EXTRACTION_SAMPLES, NUM_QUERY_SAMPLES,
    NUM_MULTIHOP_SAMPLES, NUM_RAG_SAMPLES,
)
from synsynth_io import write_json, read_json

random.seed(RANDOM_SEED)

# ============================================================================
#  辅助函数
# ============================================================================

def _cache_path(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.json")


def _cached(name: str):
    p = _cache_path(name)
    if os.path.exists(p):
        logger.info("Cache trouvé pour '%s'.", name)
        return read_json(os.path.relpath(p, safe_path(".")))
    return None


def _save_cache(name: str, data: Any):
    relpath = os.path.relpath(_cache_path(name), safe_path("."))
    write_json(relpath, data)


# ============================================================================
#  1.  关系抽取（DocRED / TACRED）
# ============================================================================

_RELATION_TYPES = [
    "founded_by", "headquarters_location", "date_of_birth",
    "country_of_citizenship", "occupation", "capital_of",
    "spouse", "employer", "award_received", "educated_at",
    "member_of", "author_of", "located_in", "part_of",
    "instance_of", "subsidiary_of", "CEO_of", "genre",
    "language_spoken", "cause_of_death",
]

_ENTITY_PAIRS = [
    ("Apple Inc.", "Steve Jobs", "founded_by"),
    ("Google", "Mountain View", "headquarters_location"),
    ("Marie Curie", "7 novembre 1867", "date_of_birth"),
    ("Albert Einstein", "Allemagne", "country_of_citizenship"),
    ("Ada Lovelace", "mathématicienne", "occupation"),
    ("Paris", "France", "capital_of"),
    ("Pierre Curie", "Marie Curie", "spouse"),
    ("Tim Cook", "Apple Inc.", "employer"),
    ("Marie Curie", "Prix Nobel de Physique", "award_received"),
    ("Alan Turing", "Université de Cambridge", "educated_at"),
]


def _generate_extraction_samples_synthetic(n: int) -> list[dict]:
    """生成句子级关系抽取样本。"""
    from synsynth_model import generate_structured

    samples: list[dict] = []
    # 先使用代码中预置的事实样本
    templates = [
        ("{head} a été fondée par {tail}.", "founded_by"),
        ("Le siège de {head} est à {tail}.", "headquarters_location"),
        ("{head} est née le {tail}.", "date_of_birth"),
        ("{head} possède la nationalité de {tail}.", "country_of_citizenship"),
        ("{head} exerçait la profession de {tail}.", "occupation"),
        ("{head} est la capitale de {tail}.", "capital_of"),
        ("{head} était marié(e) à {tail}.", "spouse"),
        ("{head} est employé(e) par {tail}.", "employer"),
        ("{head} a reçu le {tail}.", "award_received"),
        ("{head} a étudié à {tail}.", "educated_at"),
    ]
    for head, tail, rel in _ENTITY_PAIRS:
        tpl = [t for t in templates if t[1] == rel][0][0]
        samples.append({
            "text": tpl.format(head=head, tail=tail),
            "head": head,
            "tail": tail,
            "relation": rel,
        })

    # 再用模型补充样本
    if len(samples) < n:
        prompt = (
            "Génère exactement {k} exemples d'extraction de relations "
            "au format JSON (liste d'objets avec clés : text, head, tail, relation). "
            "Les relations doivent appartenir à : {rels}. "
            "Chaque objet 'text' est une phrase en français contenant head et tail."
        ).format(k=n - len(samples), rels=", ".join(_RELATION_TYPES[:10]))

        raw = generate_structured(prompt, json_mode=True, max_new_tokens=4096)
        try:
            extra = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(extra, list):
                samples.extend(extra[: n - len(samples)])
        except json.JSONDecodeError:
            logger.warning("Échec du parsing JSON pour les samples générés.")

    return samples[:n]


def _try_load_hf_extraction(n: int) -> list[dict] | None:
    """尝试从 Hugging Face 加载 Parquet 格式的 Re-DocRED。"""
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "tonytan48/Re-DocRED", split="validation",
            cache_dir=os.path.join(DATA_DIR, "hf_cache"),
        )
        logger.info("Re-DocRED chargé : %d documents.", len(ds))
        samples = []
        for ex in ds:
            if len(samples) >= n:
                break
            sents = ex.get("sents", [])
            text = " ".join(" ".join(s) for s in sents) if sents else ""
            entities = ex.get("vertexSet", [])
            for label in ex.get("labels", []):
                head_idx = label.get("h", 0)
                tail_idx = label.get("t", 0)
                relation = label.get("r", "unknown")
                if head_idx < len(entities) and tail_idx < len(entities):
                    head_name = entities[head_idx][0].get("name", "")
                    tail_name = entities[tail_idx][0].get("name", "")
                    if head_name and tail_name and head_name != tail_name:
                        samples.append({
                            "text": text[:512],
                            "head": head_name,
                            "tail": tail_name,
                            "relation": relation,
                        })
                        if len(samples) >= n:
                            break
        logger.info("Re-DocRED : %d triplets extraits.", len(samples))
        return samples if samples else None
    except Exception as e:
        logger.info("Re-DocRED HF indisponible : %s — bascule en synthétique.", e)
        return None


def load_extraction_data(n: int = NUM_EXTRACTION_SAMPLES) -> list[dict]:
    cached = _cached("extraction")
    if cached:
        return cached[:n]

    data = _try_load_hf_extraction(n) or _generate_extraction_samples_synthetic(n)
    _save_cache("extraction", data)
    return data[:n]


# ============================================================================
#  2.  WebQuestionsSP：文本到查询
# ============================================================================

_WEBQ_EXAMPLES = [
    {"question": "Quelle est la capitale de la France ?",
     "answer": "Paris",
     "cypher": "MATCH (c:Country {name:'France'})-[:HAS_CAPITAL]->(cap) RETURN cap.name"},
    {"question": "Qui a fondé Microsoft ?",
     "answer": "Bill Gates et Paul Allen",
     "cypher": "MATCH (o:Organization {name:'Microsoft'})<-[:FOUNDED]-(p) RETURN p.name"},
    {"question": "Où se trouve le siège d'Amazon ?",
     "answer": "Seattle",
     "cypher": "MATCH (o:Organization {name:'Amazon'})-[:HQ_LOCATION]->(l) RETURN l.name"},
    {"question": "Quelle est la population de Tokyo ?",
     "answer": "13,96 millions",
     "cypher": "MATCH (c:City {name:'Tokyo'}) RETURN c.population"},
    {"question": "Dans quel pays se trouve le Machu Picchu ?",
     "answer": "Pérou",
     "cypher": "MATCH (l:Landmark {name:'Machu Picchu'})-[:LOCATED_IN]->(c) RETURN c.name"},
    # --- 时间类问题 ---
    {"question": "En quelle année la Première Guerre mondiale a-t-elle commencé ?",
     "answer": "1914",
     "cypher": "MATCH (e:Event {name:'Première Guerre mondiale'}) RETURN e.start_year"},
    {"question": "Quand a été fondée l'Organisation des Nations unies ?",
     "answer": "1945",
     "cypher": "MATCH (o:Organization {name:'ONU'}) RETURN o.founded_year"},
    # --- 数值类问题 ---
    {"question": "Quelle est la superficie de l'Australie ?",
     "answer": "7,69 millions de km²",
     "cypher": "MATCH (c:Country {name:'Australie'}) RETURN c.area_km2"},
    {"question": "Combien d'habitants compte l'Inde ?",
     "answer": "1,4 milliard",
     "cypher": "MATCH (c:Country {name:'Inde'}) RETURN c.population"},
    # --- 关系类问题 ---
    {"question": "Qui est le réalisateur du film Inception ?",
     "answer": "Christopher Nolan",
     "cypher": "MATCH (f:Film {name:'Inception'})-[:DIRECTED_BY]->(p) RETURN p.name"},
    {"question": "Quel est l'auteur de 'Les Misérables' ?",
     "answer": "Victor Hugo",
     "cypher": "MATCH (b:Book {name:'Les Misérables'})-[:WRITTEN_BY]->(a) RETURN a.name"},
    {"question": "Quelle est la monnaie du Japon ?",
     "answer": "Yen",
     "cypher": "MATCH (c:Country {name:'Japon'})-[:HAS_CURRENCY]->(m) RETURN m.name"},
    # --- 地理类问题 ---
    {"question": "Quel est le plus long fleuve d'Afrique ?",
     "answer": "Le Nil",
     "cypher": "MATCH (r:River)-[:LOCATED_IN]->(cont:Continent {name:'Afrique'}) RETURN r.name ORDER BY r.length DESC LIMIT 1"},
    {"question": "Quel océan borde la côte ouest de l'Amérique du Sud ?",
     "answer": "L'océan Pacifique",
     "cypher": "MATCH (cont:Continent {name:'Amérique du Sud'})-[:BORDERED_BY]->(o:Ocean) WHERE o.side='west' RETURN o.name"},
    {"question": "Quel est le plus haut sommet du monde ?",
     "answer": "L'Everest",
     "cypher": "MATCH (m:Mountain) RETURN m.name ORDER BY m.elevation DESC LIMIT 1"},
    # --- 科学类问题 ---
    {"question": "Quel est le symbole chimique de l'or ?",
     "answer": "Au",
     "cypher": "MATCH (e:Element {name:'Or'}) RETURN e.symbol"},
    {"question": "Quelle planète est la plus proche du Soleil ?",
     "answer": "Mercure",
     "cypher": "MATCH (p:Planet) RETURN p.name ORDER BY p.distance_from_sun LIMIT 1"},
    {"question": "Qui a formulé la théorie de la relativité ?",
     "answer": "Albert Einstein",
     "cypher": "MATCH (t:Theory {name:'Relativité'})-[:FORMULATED_BY]->(p) RETURN p.name"},
    # --- 机构类问题 ---
    {"question": "Quel pays a le plus grand nombre de Prix Nobel de littérature ?",
     "answer": "La France",
     "cypher": "MATCH (p:Person)-[:WON]->(n:NobelPrize {category:'Littérature'}), (p)-[:CITIZEN_OF]->(c:Country) RETURN c.name, count(*) ORDER BY count(*) DESC LIMIT 1"},
    {"question": "Quelle est la langue officielle du Brésil ?",
     "answer": "Le portugais",
     "cypher": "MATCH (c:Country {name:'Brésil'})-[:OFFICIAL_LANGUAGE]->(l) RETURN l.name"},
]


def load_query_data(n: int = NUM_QUERY_SAMPLES) -> list[dict]:
    cached = _cached("query")
    if cached:
        return cached[:n]

    from synsynth_model import generate_structured

    samples = list(_WEBQ_EXAMPLES)
    # 持续生成，直到达到 n 个样本
    max_consecutive_failures = 5
    consecutive_failures = 0
    while len(samples) < n:
        batch_k = min(n - len(samples), 30)  # 每批最多 30 个，减少 JSON 格式错误
        prompt = (
            "Génère exactement {k} questions de type base de connaissances avec pour chaque "
            "question : question (français), answer (texte court), "
            "cypher (requête Cypher). Les questions doivent couvrir des domaines variés : "
            "géographie, histoire, science, culture, économie, sport, politique. "
            "Format JSON liste."
        ).format(k=batch_k)
        raw = generate_structured(prompt, json_mode=True, max_new_tokens=4096)
        try:
            extra = json.loads(raw) if isinstance(raw, str) else raw
            # 解包字典包裹的列表，例如 {"questions": [...]}
            if isinstance(extra, dict):
                for v in extra.values():
                    if isinstance(v, list):
                        extra = v
                        break
            if isinstance(extra, list) and extra:
                samples.extend(extra)
                consecutive_failures = 0
                logger.info("Query batch OK — %d/%d échantillons.", len(samples), n)
            else:
                consecutive_failures += 1
                logger.warning("Batch query vide (%d/%d échecs consécutifs).",
                               consecutive_failures, max_consecutive_failures)
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Trop d'échecs consécutifs — arrêt génération query.")
                    break
        except json.JSONDecodeError:
            consecutive_failures += 1
            logger.warning("JSON invalide pour query samples (%d/%d échecs consécutifs).",
                           consecutive_failures, max_consecutive_failures)
            if consecutive_failures >= max_consecutive_failures:
                logger.error("Trop d'échecs consécutifs — arrêt génération query.")
                break

    _save_cache("query", samples)
    return samples[:n]


# ============================================================================
#  3.  HotpotQA：多跳推理
# ============================================================================

_HOTPOT_EXAMPLES = [
    {
        "question": "Le fondateur de l'entreprise dont le siège est à Cupertino a étudié dans quelle université ?",
        "supporting_facts": [
            "Apple Inc. a son siège à Cupertino, Californie.",
            "Apple a été fondée par Steve Jobs.",
            "Steve Jobs a brièvement étudié au Reed College."
        ],
        "answer": "Reed College",
        "hops": 3,
    },
    {
        "question": "Quel prix Nobel a reçu la scientifique née à Varsovie qui a découvert le Polonium ?",
        "supporting_facts": [
            "Marie Curie est née à Varsovie en 1867.",
            "Marie Curie a découvert le Polonium en 1898.",
            "Marie Curie a reçu le Prix Nobel de Physique en 1903."
        ],
        "answer": "Prix Nobel de Physique",
        "hops": 3,
    },
    {
        "question": "La capitale du pays où est né Albert Einstein se situe dans quel Land ?",
        "supporting_facts": [
            "Albert Einstein est né en Allemagne (Ulm).",
            "La capitale de l'Allemagne est Berlin.",
            "Berlin est à la fois une ville et un Land (État fédéré)."
        ],
        "answer": "Berlin (Land)",
        "hops": 3,
    },
]


def _try_load_hf_hotpotqa(n: int) -> list[dict] | None:
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "hotpot_qa", "fullwiki", split="validation",
            trust_remote_code=True,
            cache_dir=os.path.join(DATA_DIR, "hf_cache"),
        )
        samples = []
        for ex in ds:
            if len(samples) >= n:
                break
            samples.append({
                "question": ex["question"],
                "answer": ex["answer"],
                "supporting_facts": [
                    t for t in ex.get("context", {}).get("sentences", [])
                ] if isinstance(ex.get("context"), dict) else [],
                "hops": 2,
            })
        return samples if samples else None
    except Exception as e:
        logger.info("HotpotQA HF indisponible : %s", e)
        return None


def load_multihop_data(n: int = NUM_MULTIHOP_SAMPLES) -> list[dict]:
    cached = _cached("multihop")
    if cached:
        return cached[:n]

    data = _try_load_hf_hotpotqa(n)
    if not data:
        from synsynth_model import generate_structured
        samples = list(_HOTPOT_EXAMPLES)
        if len(samples) < n:
            prompt = (
                "Génère {k} questions de raisonnement multi-hop en français. "
                "Chaque objet JSON doit avoir : question, supporting_facts (liste de phrases), "
                "answer, hops (entier ≥ 2). Format JSON liste."
            ).format(k=min(n - len(samples), 50))
            raw = generate_structured(prompt, json_mode=True, max_new_tokens=4096)
            try:
                extra = json.loads(raw)
                if isinstance(extra, list):
                    samples.extend(extra)
            except json.JSONDecodeError:
                pass
        data = samples

    _save_cache("multihop", data)
    return data[:n]


# ============================================================================
#  4.  RAG 与忠实度（RAGAS 风格）
# ============================================================================

_RAG_EXAMPLES = [
    {
        "question": "Quels sont les effets secondaires du Metformine ?",
        "context": (
            "Le Metformine est un antidiabétique oral de première ligne. "
            "Ses effets secondaires courants incluent des troubles gastro-intestinaux "
            "(nausées, diarrhée), et rarement une acidose lactique."
        ),
        "reference_answer": (
            "Les effets secondaires du Metformine incluent des troubles "
            "gastro-intestinaux (nausées, diarrhée) et rarement une acidose lactique."
        ),
    },
    {
        "question": "Quel article du Code Pénal français définit le vol ?",
        "context": (
            "L'article 311-1 du Code Pénal français définit le vol comme "
            "la soustraction frauduleuse de la chose d'autrui. La peine "
            "encourue est de trois ans d'emprisonnement et 45 000 euros d'amende."
        ),
        "reference_answer": (
            "L'article 311-1 du Code Pénal définit le vol."
        ),
    },
    {
        "question": "Quelle est la distance entre la Terre et le Soleil ?",
        "context": (
            "La distance moyenne entre la Terre et le Soleil est d'environ "
            "149,6 millions de kilomètres, soit une unité astronomique (UA). "
            "Cette distance varie légèrement au cours de l'année en raison "
            "de l'orbite elliptique de la Terre."
        ),
        "reference_answer": (
            "La distance moyenne est d'environ 149,6 millions de kilomètres (1 UA)."
        ),
    },
    {
        "question": "Quand la Déclaration des droits de l'homme a-t-elle été adoptée ?",
        "context": (
            "La Déclaration universelle des droits de l'homme a été adoptée "
            "par l'Assemblée générale des Nations unies le 10 décembre 1948 "
            "à Paris, au palais de Chaillot. Elle a été rédigée sous la "
            "direction d'Eleanor Roosevelt."
        ),
        "reference_answer": (
            "La Déclaration a été adoptée le 10 décembre 1948."
        ),
    },
    {
        "question": "Quel est le point d'ébullition de l'eau ?",
        "context": (
            "L'eau pure bout à 100 degrés Celsius (212 °F) à pression "
            "atmosphérique standard (1 atm). Le point d'ébullition diminue "
            "avec l'altitude en raison de la baisse de pression."
        ),
        "reference_answer": (
            "L'eau bout à 100 °C à pression atmosphérique standard."
        ),
    },
    {
        "question": "Qui a peint la Joconde ?",
        "context": (
            "La Joconde, également connue sous le nom de Mona Lisa, est "
            "un tableau peint par Léonard de Vinci entre 1503 et 1519. "
            "Il est exposé au musée du Louvre à Paris depuis 1797."
        ),
        "reference_answer": (
            "La Joconde a été peinte par Léonard de Vinci."
        ),
    },
]


def load_rag_data(n: int = NUM_RAG_SAMPLES) -> list[dict]:
    cached = _cached("rag")
    if cached:
        return cached[:n]

    from synsynth_model import generate_structured
    samples = list(_RAG_EXAMPLES)

    # 持续生成，直到达到 n 个样本
    max_consecutive_failures = 5
    consecutive_failures = 0
    while len(samples) < n:
        batch_k = min(n - len(samples), 30)
        prompt = (
            "Génère exactement {k} exemples de question-réponse avec contexte pour "
            "l'évaluation RAG. Chaque objet JSON : question, context (paragraphe factuel "
            "de 2-4 phrases), reference_answer (réponse courte). "
            "Domaines variés : médecine, droit, science, histoire, technologie, "
            "géographie, économie. Format JSON liste."
        ).format(k=batch_k)
        raw = generate_structured(prompt, json_mode=True, max_new_tokens=4096)
        try:
            extra = json.loads(raw)
            # 解包字典包裹的列表，例如 {"examples": [...]}
            if isinstance(extra, dict):
                for v in extra.values():
                    if isinstance(v, list):
                        extra = v
                        break
            if isinstance(extra, list) and extra:
                samples.extend(extra)
                consecutive_failures = 0
                logger.info("RAG batch OK — %d/%d échantillons.", len(samples), n)
            else:
                consecutive_failures += 1
                logger.warning("Batch RAG vide (%d/%d échecs consécutifs).",
                               consecutive_failures, max_consecutive_failures)
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Trop d'échecs consécutifs — arrêt génération RAG.")
                    break
        except json.JSONDecodeError:
            consecutive_failures += 1
            logger.warning("JSON invalide pour RAG samples (%d/%d échecs consécutifs).",
                           consecutive_failures, max_consecutive_failures)
            if consecutive_failures >= max_consecutive_failures:
                logger.error("Trop d'échecs consécutifs — arrêt génération RAG.")
                break

    _save_cache("rag", samples)
    return samples[:n]
