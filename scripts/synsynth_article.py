"""
根据实验结果生成论文。

Gemma-4 根据收集到的指标，撰写 Markdown / LaTeX 格式的结构化文章。
"""
from __future__ import annotations

import json
import time
from typing import Any

from synsynth_config import logger, ARTICLE_DIR, safe_path
from synsynth_model import generate
from synsynth_io import read_text, write_text


def _build_results_summary(all_results: dict[str, Any]) -> str:
    """将实验结果整理成供模型使用的文字摘要。"""
    lines = []

    ext = all_results.get("extraction", {})
    lines.append(
        f"## Extraction de relations\n"
        f"- Échantillons : {ext.get('n_samples', '?')}\n"
        f"- Précision : {ext.get('precision', '?')}\n"
        f"- Rappel : {ext.get('recall', '?')}\n"
        f"- F1-Score : {ext.get('f1_score', '?')} (cible : {ext.get('target_f1', 0.85)})\n"
        f"- Cible atteinte : {ext.get('target_met', False)}\n"
    )

    qtq = all_results.get("text_to_query", {})
    lines.append(
        f"## Text-to-Query (WebQuestionsSP style)\n"
        f"- Échantillons : {qtq.get('n_samples', '?')}\n"
        f"- Accuracy : {qtq.get('accuracy', '?')} (cible : {qtq.get('target_accuracy', 0.9)})\n"
        f"- Taux de Cypher valide : {qtq.get('cypher_syntax_valid_rate', '?')}\n"
        f"- Cible atteinte : {qtq.get('target_met', False)}\n"
    )

    mh = all_results.get("multihop_reasoning", {})
    lines.append(
        f"## Raisonnement Multi-hop (HotpotQA style)\n"
        f"- Échantillons : {mh.get('n_samples', '?')}\n"
        f"- Accuracy exacte : {mh.get('exact_accuracy', '?')}\n"
        f"- Accuracy partielle : {mh.get('partial_accuracy', '?')}\n"
        f"- Longueur moyenne de chaîne : {mh.get('avg_reasoning_chain_length', '?')}\n"
    )

    rag = all_results.get("rag_faithfulness", {})
    lines.append(
        f"## Évaluation RAGAS\n"
        f"- Échantillons : {rag.get('n_samples', '?')}\n"
        f"- Fidélité moyenne : {rag.get('avg_faithfulness', '?')} (cible ≈ 1.0)\n"
        f"- Pertinence moyenne : {rag.get('avg_answer_relevance', '?')}\n"
        f"- Précision du contexte : {rag.get('avg_context_precision', '?')}\n"
        f"- Cible atteinte : {rag.get('target_met', False)}\n"
    )

    return "\n".join(lines)


def _read_source_docs() -> str:
    """读取项目的源文档。"""
    parts = []
    try:
        parts.append("### Document 1 : Cahier des charges\n" + read_text("SYNSYNTH.md"))
    except FileNotFoundError:
        pass
    try:
        parts.append("### Document 2 : Présentation scientifique\n"
                      + read_text("Presentation_SYNSYTH.md"))
    except FileNotFoundError:
        pass
    return "\n\n".join(parts) if parts else "(documents sources non trouvés)"


ARTICLE_SECTIONS = [
    ("title_abstract", (
        "Rédige le titre et le résumé (abstract) de l'article scientifique. "
        "Le titre doit être percutant et refléter l'approche Knowledge-Graph + IA frugale. "
        "L'abstract doit faire 200-300 mots, mentionner les 4 axes d'évaluation et les "
        "résultats clés chiffrés. Rédige en français académique."
    )),
    ("introduction", (
        "Rédige l'introduction (section 1). "
        "Présente le contexte (LLM, hallucinations, coût environnemental), "
        "la problématique (comment coupler KG et LLM frugal), "
        "les contributions du papier (4 axes d'évaluation), "
        "et le plan de l'article. 600-800 mots. Français académique."
    )),
    ("related_work", (
        "Rédige l'état de l'art (section 2). "
        "Couvre : (a) Construction de KG (pipelines classiques vs génératifs), "
        "(b) Extraction de relations documentaire (DocRED, ATLOP, DREEAM), "
        "(c) Text-to-Query et raisonnement multi-hop, "
        "(d) RAG et évaluation anti-hallucination (RAGAS). "
        "Cite les travaux mentionnés dans les documents sources. 800-1000 mots."
    )),
    ("methodology", (
        "Rédige la méthodologie (section 3). Décris : "
        "(a) L'architecture SYNSYNTH+ (4 phases), "
        "(b) Le modèle utilisé (Gemma-4-26B-A4B-it, quantification GGUF UD-Q4_K_XK via unsloth), "
        "(c) Le protocole expérimental pour chaque axe, "
        "(d) Les métriques et benchmarks. 800-1000 mots."
    )),
    ("results", (
        "Rédige la section résultats (section 4). "
        "Présente les résultats de chaque expérience avec des tableaux en LaTeX/Markdown. "
        "Compare aux baselines connues (DREEAM 80.20%, ATLOP 77.81%). "
        "Analyse les forces et faiblesses. 800-1000 mots."
    )),
    ("discussion", (
        "Rédige la discussion (section 5). "
        "Analyse : (a) La frugalité du modèle quantifié vs modèles pleins, "
        "(b) Le trade-off performance/empreinte carbone, "
        "(c) Les limites (données synthétiques, taille d'évaluation), "
        "(d) Perspectives pour SYNSYNTH+ Phase 4 (domaines verticaux). 600-800 mots."
    )),
    ("conclusion", (
        "Rédige la conclusion (section 6). "
        "Résume les contributions, les résultats principaux, "
        "et les perspectives futures. 300-400 mots."
    )),
    ("references", (
        "Rédige la section références. "
        "Inclus les références académiques pertinentes mentionnées dans les documents "
        "et dans l'état de l'art : DocRED, TACRED, ATLOP, DREEAM, HotpotQA, "
        "WebQuestionsSP, RAGAS, unsloth, Gemma, etc. Format BibTeX ou numéroté."
    )),
]


def generate_article(all_results: dict[str, Any]) -> str:
    """逐节生成完整文章。"""
    logger.info("=== Génération de l'article scientifique ===")

    results_summary = _build_results_summary(all_results)
    source_docs = _read_source_docs()

    base_system = (
        "Tu es un chercheur senior en intelligence artificielle spécialisé dans "
        "les graphes de connaissances et l'IA frugale. Tu rédiges un article "
        "scientifique de haut niveau pour une conférence internationale (ACL, EMNLP, "
        "COLING). Le style est académique, rigoureux, en français. "
        "Tu utilises des formulations précises et des références quand approprié. "
        "Tu inclus des tableaux et formules quand pertinent."
    )

    context_block = (
        f"=== RÉSULTATS EXPÉRIMENTAUX ===\n{results_summary}\n\n"
        f"=== DOCUMENTS SOURCES DU PROJET ===\n{source_docs[:4000]}\n"
    )

    article_parts: list[str] = []
    t0 = time.time()

    for section_id, instruction in ARTICLE_SECTIONS:
        logger.info("  Rédaction : %s …", section_id)
        prompt = (
            f"{context_block}\n\n"
            f"=== INSTRUCTION ===\n{instruction}\n\n"
            "Rédige UNIQUEMENT cette section, en Markdown. "
            "Utilise des sous-sections (##, ###) appropriées. "
            "N'ajoute pas les sections précédentes ou suivantes."
        )
        section_text = generate(
            prompt,
            system=base_system,
            max_new_tokens=4096,
            temperature=0.4,
        )
        article_parts.append(section_text)
        logger.info("  ✓ %s — %d car.", section_id, len(section_text))

    elapsed = time.time() - t0
    logger.info("Article complet généré en %.1fs.", elapsed)

    # 合并各节内容
    full_article = "\n\n---\n\n".join(article_parts)

    # 保存文章
    write_text("article/SYNSYNTH_article.md", full_article)

    # 同时将原始结果保存为 JSON
    results_json = json.dumps(all_results, ensure_ascii=False, indent=2, default=str)
    write_text("results/all_results.json", results_json)

    return full_article
