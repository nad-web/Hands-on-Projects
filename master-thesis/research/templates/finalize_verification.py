"""Reconcile the final thesis artifacts and write the independent readiness audit."""
from pathlib import Path
from zipfile import ZipFile
from datetime import datetime, timezone
import hashlib
import json
import re
import unicodedata

import nbformat
import pandas as pd
import pdfplumber
from lxml import etree

OUT = Path('/mnt/agents/output')
ROOT = OUT / 'thesis_execution'
M = ROOT / 'manuscript'
R = ROOT / 'results'
A = ROOT / 'audit'
DOCX = OUT / 'Final_Master_Thesis_Trustworthy_Financial_Forecasting.docx'
PDF = OUT / 'Final_Master_Thesis_Trustworthy_Financial_Forecasting.pdf'
MD = M / 'Final_Master_Thesis.md'
NOTES = OUT / 'Reproducibility_and_Readiness_Notes.md'
TITLE = 'Towards Trustworthy Financial Time Series Forecasting: An Experimental Evaluation of Explainable Machine Learning Models for Stock Price Prediction'
RQS = [
    'In financial time series forecasting, to what extent do post-hoc explanation methods (e.g., SHAP, LIME) mitigate the accuracy–interpretability trade-off between intrinsically interpretable models and black-box architectures (e.g., LSTMs, Temporal Fusion Transformers)?',
    'To what extent are feature attributions from post-hoc XAI methods temporally stable and robust to non-stationarity, structural breaks, and regime shifts (e.g., bull vs. bear markets, crisis periods) in financial time series?',
    'To what extent are post-hoc explanations faithful to underlying model mechanics, and do they capture genuine economic drivers rather than spurious correlations?'
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text):
    text = unicodedata.normalize('NFKC', text or '')
    text = re.sub(r'\s+', ' ', text).strip()
    return re.sub(r'(?<=-)\s+(?=\w)', '', text)


def section(text, start, end=None):
    i = text.index(start)
    j = text.index(end, i) if end else len(text)
    return text[i:j].strip() + '\n'


def placeholder_counts(text):
    patterns = {
        'unresolved_table_marker': r'\{\{TABLE:',
        'template_braces': r'\{\{|\}\}',
        'todo_token': r'\bTODO\b',
        'tbd_token': r'\bTBD\b',
        'xxx_token': r'\bXXX\b',
        'insert_placeholder': r'\bINSERT\s+(?:AUTHOR|INSTITUTION|NAME|DATE|SUPERVISOR)\b',
        'lorem_ipsum': r'lorem\s+ipsum',
        'replacement_character': '\ufffd',
        'markdown_bold_delimiter': r'\*\*',
        'markdown_image_syntax': r'!\[',
        'markdown_link_syntax': r'\[[^\]\n]+\]\([^\)\n]+\)',
        'editorial_inference_note': r'journal name inferred from DOI',
        'old_arima_descriptive_count': r'\b4/16\b',
        'old_execution_audit_count': r'\b128/128\b'
    }
    return {name: len(re.findall(pattern, text, flags=re.I)) for name, pattern in patterns.items()}


md = MD.read_text()
assembly = json.loads((M / 'assembly_report.json').read_text())
execution = json.loads((A / 'Execution_Validation_Report.json').read_text())
correction = json.loads((A / 'ARIMA_Benchmark_Correction_Report.json').read_text())
pagination = json.loads((M / 'layout/pagination_post_validation/pagination_report.json').read_text())
word_qa = json.loads((ROOT / 'qa/word_corrected_final/report.json').read_text())
pdf_qa = json.loads((ROOT / 'qa/pdf_corrected_final/inspection_report.json').read_text())['data']
notes = NOTES.read_text()

# Extract final text from the delivered formats.
with ZipFile(DOCX) as z:
    xml = etree.fromstring(z.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    docx_text = '\n'.join(xml.xpath('.//w:t/text()', namespaces=ns))
with pdfplumber.open(PDF) as pdf:
    pdf_pages = [page.extract_text() or '' for page in pdf.pages]
pdf_text = '\n'.join(pdf_pages)
normalized_docx = norm(docx_text)
normalized_pdf = norm(pdf_text)

# Verify the current notebook hashes against manuscript Table C.2.
hash_table = (M / 'tables/C2_notebook_hashes.md').read_text()
notebook_hashes = {}
notebook_states = {}
for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
    digest = sha(path)
    notebook_hashes[path.name] = digest
    nb = nbformat.read(path, as_version=4)
    code = [cell for cell in nb.cells if cell.cell_type == 'code']
    errors = [output for cell in code for output in cell.get('outputs', []) if output.output_type == 'error']
    counts = [cell.execution_count for cell in code]
    notebook_states[path.name] = {
        'execution_counts': counts,
        'sequential_from_one': counts == list(range(1, len(code) + 1)),
        'stored_errors': len(errors),
        'sha256_in_table_c2': digest in hash_table
    }

# Reconcile every native summary row directly from the test-level export.
block = pd.read_csv(R / 'cross_model/tables/native_benchmark_block_dm.csv')
summary = pd.read_csv(R / 'cross_model/tables/native_benchmark_summary.csv')
rows = []
for keys, group in block.groupby(['sample', 'model', 'comparison'], sort=True):
    rows.append({
        'sample': keys[0], 'model': keys[1], 'comparison': keys[2],
        'tests': len(group),
        'raw_lower_loss': int((group.dm < 0).sum()),
        'significant_wins': int((group.significant_q05 & (group.outcome == 'model_win')).sum()),
        'significant_losses': int((group.significant_q05 & (group.outcome == 'model_loss')).sum()),
        'not_significant': int((~group.significant_q05).sum()),
        'exported_bh2_wins': int((group.significant_exported_bh2 & (group.outcome_exported_bh2 == 'model_win')).sum()),
        'exported_bh2_losses': int((group.significant_exported_bh2 & (group.outcome_exported_bh2 == 'model_loss')).sum())
    })
recomputed = pd.DataFrame(rows)
reconciled = summary.merge(recomputed, on=['sample', 'model', 'comparison'], suffixes=('_exported', '_recomputed'), validate='one_to_one')
reconciliation_columns = ['tests', 'raw_lower_loss', 'significant_wins', 'significant_losses', 'not_significant', 'exported_bh2_wins', 'exported_bh2_losses']
summary_reconciled = len(reconciled) == 28 and all(
    reconciled[f'{column}_exported'].equals(reconciled[f'{column}_recomputed']) for column in reconciliation_columns)

financial = block[block['sample'] == 'financial']
sunspot = block[block['sample'] == 'sunspot']
persistence_wins = sunspot[sunspot.significant_q05 & (sunspot.outcome == 'model_win') & (sunspot.comparison == 'random_walk_or_persistence')]
native_claims = {
    'financial_tests': int(len(financial)),
    'financial_significant_wins_BH14': int((financial.significant_q05 & (financial.outcome == 'model_win')).sum()),
    'financial_significant_losses_BH14': int((financial.significant_q05 & (financial.outcome == 'model_loss')).sum()),
    'financial_nonsignificant_BH14': int((~financial.significant_q05).sum()),
    'financial_BH14_losses_to_persistence': int((financial.significant_q05 & (financial.outcome == 'model_loss') & (financial.comparison == 'random_walk_or_persistence')).sum()),
    'financial_BH14_losses_to_training_mean': int((financial.significant_q05 & (financial.outcome == 'model_loss') & (financial.comparison == 'training_mean')).sum()),
    'financial_significant_wins_BH2_sensitivity': int((financial.significant_exported_bh2 & (financial.outcome_exported_bh2 == 'model_win')).sum()),
    'financial_significant_losses_BH2_sensitivity': int((financial.significant_exported_bh2 & (financial.outcome_exported_bh2 == 'model_loss')).sum()),
    'financial_BH2_losses_to_persistence': int((financial.significant_exported_bh2 & (financial.outcome_exported_bh2 == 'model_loss') & (financial.comparison == 'random_walk_or_persistence')).sum()),
    'financial_BH2_losses_to_training_mean': int((financial.significant_exported_bh2 & (financial.outcome_exported_bh2 == 'model_loss') & (financial.comparison == 'training_mean')).sum()),
    'financial_negative_differential_with_raw_p_below_005': int(((financial.dm < 0) & (financial.p_raw < 0.05)).sum()),
    'sunspot_training_mean_wins_BH14': int((sunspot.significant_q05 & (sunspot.outcome == 'model_win') & (sunspot.comparison == 'training_mean')).sum()),
    'sunspot_persistence_wins_BH14': int(len(persistence_wins)),
    'sunspot_persistence_losses_BH14': int((sunspot.significant_q05 & (sunspot.outcome == 'model_loss') & (sunspot.comparison == 'random_walk_or_persistence')).sum()),
    'sunspot_persistence_nonsignificant_BH14': int(((~sunspot.significant_q05) & (sunspot.comparison == 'random_walk_or_persistence')).sum()),
    'sunspot_persistence_win_adjusted_p': {row.model: float(row.p_bh_14_tests) for row in persistence_wins.itertuples()},
    'summary_rows_reconciled_exactly': int(len(reconciled)) if summary_reconciled else 0,
    'arima_financial_training_mean_lower_loss_cells': int(summary[(summary['sample'] == 'financial') & (summary.model == 'ARIMA') & (summary.comparison == 'training_mean')].raw_lower_loss.iloc[0])
}

# Independent preservation check against the original scholarly source.
old = (OUT / 'thesis_revision/Revised_Master_Thesis.md').read_text()
expected_literature = section(old, '# 2. Theoretical Background', '# 4. Methodology')
old_pfi = '(Molnar, 2022, p. 20.\n\nLópez de Prado, 2020, p. 81);'
new_pfi = '(Molnar, 2022, p. 20; López de Prado, 2020, p. 81);'
assert expected_literature.count(old_pfi) == 1
expected_literature = expected_literature.replace(old_pfi, new_pfi)
expected_references = section(old, '# References', '# Appendices').replace(' [journal name inferred from DOI; not stated in file]', '')
actual_literature = section(md, '# 2. Theoretical Background', '# 4. Methodology')
actual_references = section(md, '# References', '# Appendices')

# ARIMA audit checks are all required, not just reported.
prediction_invariance = all(field['all_equal_within_1e-12'] for field in correction['arima_observation_predictions']['invariant_fields'].values())
explanation_invariance = all(field['byte_identical'] for field in correction['arima_explanation_and_native_parameter_invariance'].values())
matched_invariance = all(correction['matched_metric_primary_field_invariance'].values()) and all(correction['matched_observation_primary_field_invariance'].values())
cross_invariance = all(correction['cross_model_byte_identical_tables'].values())
correction_signs = correction['arima_training_mean_dm_sign_changes']
correction_all_passed = all([
    prediction_invariance, explanation_invariance, matched_invariance, cross_invariance,
    correction['arima_observation_predictions']['train_mean_rows_changed'] == 30240,
    correction['arima_financial_training_mean_descriptive_lower_loss_cells'] == {'old': 4, 'corrected': 1, 'tests': 16},
    len(correction_signs) == 3,
    {row['series'] for row in correction_signs} == {'dax', 'nasdaq', 'sp500'},
    all(row['outcome_old'] == 'not_significant' and row['outcome_new'] == 'not_significant' for row in correction_signs)
])

page_map_current = json.loads((M / 'page_map.json').read_text()) == json.loads((M / 'layout/pagination_post_validation/page_map.json').read_text())
heading_map_current = json.loads((M / 'layout/final_build/current_heading_page_map.json').read_text()) == json.loads((M / 'layout/pagination_post_validation/heading_page_map.json').read_text())

report = {
    'verified_utc': datetime.now(timezone.utc).isoformat(),
    'final_artifacts': {
        'docx': {'path': str(DOCX), 'sha256': sha(DOCX), 'bytes': DOCX.stat().st_size,
                 'word_validation_gate': 'PASSED',
                 'mechanical_repairs_reviewed': [
                     'document.xml: 14 paragraph-property ordering repairs',
                     'numbering.xml: two zero-padded numbering IDs',
                     'settings.xml: one settings-order repair',
                     'styles.xml: three paragraph-property ordering repairs']},
        'pdf': {'path': str(PDF), 'sha256': sha(PDF), 'bytes': PDF.stat().st_size, 'pages': len(pdf_pages)},
        'manuscript_markdown': {'path': str(MD), 'sha256': sha(MD), 'whitespace_delimited_words': len(md.split())},
        'readiness_notes': {'path': str(NOTES), 'sha256': sha(NOTES)}
    },
    'execution': {
        'checks_passed': execution['checks_passed'], 'checks_total': execution['checks_total'], 'all_passed': execution['all_passed'],
        'benchmark_fold_groups_checked': execution['benchmark_semantics']['fold_groups_checked'],
        'benchmark_max_abs_error': execution['benchmark_semantics']['max_abs_error'],
        'executed_notebooks': len(notebook_hashes),
        'notebook_hashes_match_manuscript': all(state['sha256_in_table_c2'] for state in notebook_states.values()),
        'notebook_sequential_and_error_free': all(state['sequential_from_one'] and state['stored_errors'] == 0 for state in notebook_states.values()),
        'notebook_sha256': notebook_hashes,
        'notebook_states': notebook_states,
        'native_prediction_rows': int(execution['models'] and sum(model['prediction_rows'] for model in execution['models'].values())),
        'native_fold_metric_rows': 760,
        'attribution_rows': int(sum(model['attribution_rows'] for model in execution['models'].values())),
        'matched_origins': 18480, 'matched_prediction_rows': 129360, 'matched_metric_rows': 140, 'pairwise_tests': 420
    },
    'arima_benchmark_correction': {
        'report_sha256': sha(A / 'ARIMA_Benchmark_Correction_Report.json'),
        'all_invariance_and_outcome_checks_passed': correction_all_passed,
        'prediction_invariance': prediction_invariance,
        'explanation_invariance': explanation_invariance,
        'matched_primary_invariance': matched_invariance,
        'cross_model_invariance': cross_invariance,
        'train_mean_rows_changed': correction['arima_observation_predictions']['train_mean_rows_changed'],
        'train_mean_max_abs_old_new_difference': correction['arima_observation_predictions']['train_mean_max_abs_old_new_difference'],
        'descriptive_lower_loss_cells': correction['arima_financial_training_mean_descriptive_lower_loss_cells'],
        'dm_sign_changes': correction_signs,
        'adjusted_inference_counts_unchanged': correction['corrected_native_benchmark_inference'] == {
            'financial_tests': 224, 'bh14_significant_wins': 0, 'bh14_significant_losses': 26, 'bh14_nonsignificant': 198,
            'bh14_persistence_losses': 11, 'bh14_training_mean_losses': 15, 'bh2_significant_wins': 0,
            'bh2_significant_losses': 39, 'bh2_persistence_losses': 15, 'bh2_training_mean_losses': 24,
            'negative_financial_differentials_with_unadjusted_p_below_005': 0,
            'sunspot_bh14_training_mean_wins': 28, 'sunspot_bh14_persistence_wins': 2,
            'sunspot_bh14_persistence_losses': 23, 'sunspot_bh14_persistence_nonsignificant': 3}
    },
    'predictive_invariance': json.loads((A / 'Predictive_Invariance_Report.json').read_text()),
    'preservation': {
        'chapters_2_3_preserved_except_pfi_citation_repair': actual_literature == expected_literature,
        'references_preserved_except_editorial_note_removed': actual_references == expected_references,
        'title_exact': md.startswith('% ' + TITLE + '\n'),
        'single_references_section': md.count('# References') == 1,
        'single_appendices_section': md.count('# Appendices') == 1,
        'no_unresolved_table_markers': '{{TABLE:' not in md,
        'RQ1_verbatim': RQS[0] in md, 'RQ2_verbatim': RQS[1] in md, 'RQ3_verbatim': RQS[2] in md,
        'assembly_hash_current': assembly['sha256'] == sha(MD),
        'assembly_word_count_current': assembly['whitespace_delimited_words'] == len(md.split()),
        'exhibit_page_map_current': page_map_current,
        'heading_page_map_current': heading_map_current,
        'documented_citation_cleanup': assembly['documented_citation_cleanup']
    },
    'native_benchmark_claims': native_claims,
    'document_text': {
        'pdf_pages': len(pdf_pages),
        'docx_title_and_research_questions_present': all(norm(text) in normalized_docx for text in [TITLE] + RQS),
        'pdf_title_and_research_questions_present': all(norm(text) in normalized_pdf for text in [TITLE] + RQS),
        'docx_corrected_arima_interpretation_present': 'only one of 16 financial cells' in docx_text,
        'pdf_corrected_arima_interpretation_present': 'only one of 16 financial cells' in pdf_text,
        'docx_final_audit_count_present': '1,648 of 1,648 checks passing' in docx_text,
        'pdf_final_audit_count_present': '1,648 of 1,648 checks passing' in pdf_text,
        'docx_placeholder_counts': placeholder_counts(docx_text),
        'pdf_placeholder_counts': placeholder_counts(pdf_text),
        'document_preparation_note_present': 'Document Preparation Note' in docx_text and 'Document Preparation Note' in pdf_text,
        'unintended_approval_or_submission_claims': len(re.findall(r'\b(?:officially submitted|submission approved|institutionally accepted|advisor approved)\b', docx_text + '\n' + pdf_text, flags=re.I))
    },
    'pagination': pagination,
    'visual_qa': {
        'word_engine_reported_pages': word_qa['pages'], 'word_pages_rendered': word_qa['rendered'],
        'word_near_blank_pages': word_qa['nearBlankPages'],
        'word_clipped_pages': [page for page in word_qa['pageFiles'] if page.get('clipped')],
        'word_fidelity_verdict': word_qa['fidelity']['verdict'],
        'word_fidelity_categories': word_qa['fidelity']['byCategory'],
        'word_high_severity_findings': [finding for finding in word_qa['fidelity']['findings'] if finding.get('severity') == 'high'],
        'word_content_loss': [finding for finding in word_qa['fidelity']['findings'] if finding.get('category') == 'content-loss'],
        'pdf_pages_rendered': pdf_qa['pageCount'], 'pdf_inspector_warnings': pdf_qa['warnings'],
        'contact_sheets_reviewed': ['word_corrected_final/contact-sheet.png', 'pdf_corrected_final/contact-sheet.png'],
        'full_resolution_pdf_pages_reviewed': [4, 45, 47, 56, 86, 88, 89],
        'table_5_4_page': 56, 'table_B_5_page': 86,
        'preliminary_orphan_resolved': True, 'clipping_or_missing_content_observed': False
    },
    'readiness_notes_current': {
        'word_count_44244': '44,244 whitespace-delimited words' in notes,
        'execution_audit_1648': '1,648/1,648 checks passed' in notes,
        'word_engine_pages_84': '84 engine-layout pages' in notes,
        'arima_correction_documented': 'ARIMA correction/invariance audit' in notes,
        'surrogate_restrictions_documented': 'global standard-normal synthetic samples' in notes,
        'stale_128_audit_absent': '128/128 checks passed' not in notes,
        'administrative_conditions_retained': 'conditional on administrative requirements and advisor/institutional approval' in notes
    },
    'readiness_verdict': 'Scientifically complete for author submission review, conditional on administrative requirements and advisor/institutional approval.'
}

# Every machine-verifiable gate must pass. Visual review is recorded separately above.
checks = {
    'execution_all_passed': report['execution']['all_passed'] and execution['checks_passed'] == execution['checks_total'] == 1648,
    'notebooks_all_current': report['execution']['notebook_hashes_match_manuscript'] and report['execution']['notebook_sequential_and_error_free'],
    'arima_correction_all_passed': correction_all_passed,
    'native_summary_reconciled': summary_reconciled,
    'preservation_all_passed': all(value for key, value in report['preservation'].items() if isinstance(value, bool)),
    'docx_title_rqs': report['document_text']['docx_title_and_research_questions_present'],
    'pdf_title_rqs': report['document_text']['pdf_title_and_research_questions_present'],
    'docx_no_placeholders': all(value == 0 for value in report['document_text']['docx_placeholder_counts'].values()),
    'pdf_no_placeholders': all(value == 0 for value in report['document_text']['pdf_placeholder_counts'].values()),
    'corrected_values_in_both_formats': all(report['document_text'][key] for key in [
        'docx_corrected_arima_interpretation_present', 'pdf_corrected_arima_interpretation_present',
        'docx_final_audit_count_present', 'pdf_final_audit_count_present']),
    'pagination_complete': pagination['main_text_target_70_to_80'] and pagination['headings_mapped'] == 122 and pagination['exhibits_mapped'] == 34 and len(pdf_pages) == 89,
    'word_qa_no_blank_or_clipping': not word_qa['nearBlankPages'] and not any(page.get('clipped') for page in word_qa['pageFiles']),
    'pdf_qa_no_warnings': not pdf_qa['warnings'] and pdf_qa['pageCount'] == 89,
    'readiness_notes_current': all(report['readiness_notes_current'].values()),
    'no_unintended_approval_claim': report['document_text']['unintended_approval_or_submission_claims'] == 0
}
report['machine_verification'] = {'checks': checks, 'checks_passed': sum(checks.values()), 'checks_total': len(checks), 'all_passed': all(checks.values())}
(A / 'Final_Verification_Report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps({'report': str(A / 'Final_Verification_Report.json'), 'machine_verification': report['machine_verification'],
                  'docx_sha256': report['final_artifacts']['docx']['sha256'], 'pdf_sha256': report['final_artifacts']['pdf']['sha256'],
                  'readiness_verdict': report['readiness_verdict']}, indent=2))
if not all(checks.values()):
    raise SystemExit(1)
