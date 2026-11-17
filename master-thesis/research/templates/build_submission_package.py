#!/usr/bin/env python3
"""Build and verify the corrected final Master thesis submission archive."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import zipfile

OUTPUT = Path('/mnt/agents/output')
ROOT = OUTPUT / 'thesis_execution'
DOCX = OUTPUT / 'Final_Master_Thesis_Trustworthy_Financial_Forecasting.docx'
PDF = OUTPUT / 'Final_Master_Thesis_Trustworthy_Financial_Forecasting.pdf'
NOTES = OUTPUT / 'Reproducibility_and_Readiness_Notes.md'
ARCHIVE = OUTPUT / 'Final_Master_Thesis_Submission_Package.zip'
EXTERNAL_SUMS = OUTPUT / 'Final_Master_Thesis_Deliverable_Checksums.sha256'
VERIFICATION = ROOT / 'audit' / 'Submission_Package_Verification.json'
FINAL_REPORT = ROOT / 'audit' / 'Final_Verification_Report.json'
EXECUTION_REPORT = ROOT / 'audit' / 'Execution_Validation_Report.json'
CORRECTION_REPORT = ROOT / 'audit' / 'ARIMA_Benchmark_Correction_Report.json'
READINESS_VERDICT = (
    'Scientifically complete for author submission review, conditional on '
    'administrative requirements and advisor/institutional approval.'
)
INVENTORY_SCOPE = (
    'file_records covers payload files; PACKAGE_MANIFEST.json and '
    'SHA256SUMS.txt are added separately. SHA256SUMS.txt covers payload '
    'plus PACKAGE_MANIFEST.json.'
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def add_file(payload: dict[str, Path], archive_path: str, source: Path) -> None:
    path = PurePosixPath(archive_path)
    assert not path.is_absolute() and '..' not in path.parts
    assert source.is_file(), f'Missing required payload file: {source}'
    assert archive_path not in payload, f'Duplicate archive path: {archive_path}'
    assert source.name != '.DS_Store'
    assert 'pre_arima_benchmark_correction' not in path.parts
    payload[archive_path] = source


def add_tree(payload: dict[str, Path], source_dir: Path, archive_dir: str) -> None:
    assert source_dir.is_dir(), f'Missing required payload directory: {source_dir}'
    for source in sorted(source_dir.rglob('*')):
        if not source.is_file() or source.name == '.DS_Store':
            continue
        relative = source.relative_to(source_dir).as_posix()
        add_file(payload, f'{archive_dir}/{relative}', source)


def collect_payload() -> dict[str, Path]:
    payload: dict[str, Path] = {}
    add_file(payload, f'thesis/{DOCX.name}', DOCX)
    add_file(payload, f'thesis/{PDF.name}', PDF)
    add_file(payload, NOTES.name, NOTES)

    add_tree(payload, ROOT / 'notebooks', 'notebooks')
    add_tree(payload, ROOT / 'results', 'results')
    add_tree(payload, ROOT / 'data' / 'source_snapshot' / 'inputs',
             'data/source_snapshot/inputs')
    add_tree(payload, ROOT / 'templates', 'templates')

    manuscript = ROOT / 'manuscript'
    for source in sorted(manuscript.iterdir()):
        if source.is_file() and source.suffix in {'.md', '.py', '.json'}:
            add_file(payload, f'manuscript/{source.name}', source)
    add_tree(payload, manuscript / 'tables', 'manuscript/tables')
    add_tree(payload, manuscript / 'figures', 'manuscript/figures')

    layout = manuscript / 'layout'
    for name in (
        'build_final_documents.py', 'derive_page_map.py', 'finalize_docx.py',
        'normalize_ooxml.py', 'thesis-reference.docx',
        'final_build/build_report.json',
        'final_build/current_heading_page_map.json',
    ):
        add_file(payload, f'manuscript/layout/{name}', layout / name)
    for directory in ('pagination_corrected_final', 'pagination_post_validation'):
        add_tree(payload, layout / directory, f'manuscript/layout/{directory}')

    for name in (
        'Execution_Validation_Report.json', 'Execution_Validation_Report.md',
        'Final_Verification_Report.json', 'Predictive_Invariance_Report.json',
        'ARIMA_Benchmark_Correction_Report.json',
        'ARIMA_Benchmark_Correction_Report.md',
        'ARIMA_Benchmark_DM_Sign_Changes.csv',
        'build_arima_correction_report.py',
    ):
        add_file(payload, f'audit/{name}', ROOT / 'audit' / name)
    add_tree(payload, ROOT / 'audit' / 'pre_regime_span', 'audit/pre_regime_span')

    word_qa = ROOT / 'qa' / 'word_corrected_final'
    for name in ('report.json', 'contact-sheet.png', 'page-04.png'):
        add_file(payload, f'qa/word_corrected_final/{name}', word_qa / name)
    pdf_qa = ROOT / 'qa' / 'pdf_corrected_final'
    for name in ('inspection_report.json', 'contact-sheet.png'):
        add_file(payload, f'qa/pdf_corrected_final/{name}', pdf_qa / name)
    for page in (4, 45, 47, 56, 86, 88, 89):
        name = f'page-{page:03d}.png'
        add_file(payload, f'qa/pdf_corrected_final/{name}', pdf_qa / name)

    add_tree(payload, ROOT / 'source', 'source')
    revised = OUTPUT / 'thesis_revision' / 'Revised_Master_Thesis.md'
    add_file(payload, 'source/Revised_Master_Thesis.md', revised)
    return payload


def validate_sources(payload: dict[str, Path], final: dict, execution: dict,
                     correction: dict) -> None:
    assert final['machine_verification']['all_passed'] is True
    assert final['machine_verification']['checks_passed'] == 15
    assert final['machine_verification']['checks_total'] == 15
    assert final['execution']['all_passed'] is True
    assert final['execution']['checks_passed'] == 1648
    assert final['execution']['checks_total'] == 1648
    assert execution['all_passed'] is True
    assert execution['checks_passed'] == execution['checks_total'] == 1648
    assert correction['independent_execution_validation']['all_passed'] is True
    assert final['arima_benchmark_correction']['all_invariance_and_outcome_checks_passed'] is True

    assert sha256_file(DOCX) == final['final_artifacts']['docx']['sha256']
    assert sha256_file(PDF) == final['final_artifacts']['pdf']['sha256']
    assert sha256_file(NOTES) == final['final_artifacts']['readiness_notes']['sha256']
    manuscript = ROOT / 'manuscript' / 'Final_Master_Thesis.md'
    assert sha256_file(manuscript) == final['final_artifacts']['manuscript_markdown']['sha256']

    notebooks = sorted(path for path in payload if path.startswith('notebooks/'))
    assert len(notebooks) == 8
    for archive_path in notebooks:
        name = PurePosixPath(archive_path).name
        expected = final['execution']['notebook_sha256'][name]
        assert sha256_file(payload[archive_path]) == expected

    assert not any('pre_arima_benchmark_correction' in path for path in payload)
    assert not any(PurePosixPath(path).name == '.DS_Store' for path in payload)
    assert not any(path.startswith('qa/word_final') for path in payload)
    assert not any(path.startswith('qa/pdf_final') for path in payload)
    assert not any(path.endswith(('.docx', '.pdf')) and
                   path.startswith('manuscript/layout/final_build/') for path in payload)


def parse_sums(text: str) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, path = line.split('  ', 1)
        assert len(digest) == 64 and path not in records
        records[path] = digest
    return records


def verify_zip(path: Path, payload: dict[str, Path]) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), 'Duplicate ZIP member names'
        expected_names = set(payload) | {'PACKAGE_MANIFEST.json', 'SHA256SUMS.txt'}
        assert set(names) == expected_names
        assert not any(name.endswith('/') for name in names)
        assert not any('pre_arima_benchmark_correction' in name for name in names)
        assert not any(PurePosixPath(name).name == '.DS_Store' for name in names)

        manifest_bytes = archive.read('PACKAGE_MANIFEST.json')
        manifest = json.loads(manifest_bytes)
        sums_text = archive.read('SHA256SUMS.txt').decode('utf-8')
        sums = parse_sums(sums_text)
        assert set(sums) == expected_names - {'SHA256SUMS.txt'}
        assert sums['PACKAGE_MANIFEST.json'] == sha256_bytes(manifest_bytes)

        # Read every member to EOF once. ZipExtFile validates each member's CRC
        # at EOF; the same pass verifies every checksum without redundant I/O.
        archived_hashes = {}
        for name in names:
            digest = hashlib.sha256()
            with archive.open(name) as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            archived_hashes[name] = digest.hexdigest()
        for name, expected in sums.items():
            assert archived_hashes[name] == expected, f'Archived checksum mismatch: {name}'

        manifest_records = {record['path']: record for record in manifest['file_records']}
        assert set(manifest_records) == set(payload)
        for name, record in manifest_records.items():
            assert archive.getinfo(name).file_size == record['bytes']
            assert archived_hashes[name] == record['sha256']
            assert sums[name] == record['sha256']

        assert manifest['payload_file_count'] == len(payload)
        assert manifest['archive_file_count_including_manifest_and_checksums'] == len(names)
        assert manifest['execution_checks'] == '1648/1648 passed'
        assert manifest['final_cross_artifact_checks'] == '15/15 passed'
        assert manifest['readiness_verdict'] == READINESS_VERDICT
        assert manifest['final_docx_sha256'] == sha256_file(DOCX)
        assert manifest['final_pdf_sha256'] == sha256_file(PDF)
        assert archived_hashes[f'thesis/{DOCX.name}'] == sha256_file(DOCX)
        assert archived_hashes[f'thesis/{PDF.name}'] == sha256_file(PDF)
        for name, expected in manifest['notebook_sha256'].items():
            assert archived_hashes[f'notebooks/{name}'] == expected

        return {
            'zip_crc_validation': 'PASSED — every member read to EOF and CRC checked',
            'archived_checksums_verified': len(sums),
            'payload_manifest_records_match_archive': True,
            'manifest_file_count_matches': True,
            'current_docx_pdf_and_notebook_hashes_match': True,
            'superseded_pre_correction_evidence_excluded': True,
            'obsolete_document_builds_and_old_qa_excluded': True,
            'payload_file_count': manifest['payload_file_count'],
            'archive_file_count_including_manifest_and_checksums': len(names),
            'payload_uncompressed_bytes': manifest['payload_uncompressed_bytes'],
            'archive_compressed_bytes': path.stat().st_size,
        }


def main() -> None:
    final = json.loads(FINAL_REPORT.read_text(encoding='utf-8'))
    execution = json.loads(EXECUTION_REPORT.read_text(encoding='utf-8'))
    correction = json.loads(CORRECTION_REPORT.read_text(encoding='utf-8'))
    payload = collect_payload()
    validate_sources(payload, final, execution, correction)

    records = []
    for archive_path in sorted(payload):
        source = payload[archive_path]
        records.append({
            'path': archive_path,
            'bytes': source.stat().st_size,
            'sha256': sha256_file(source),
        })
    manifest = {
        'package': 'Final Master Thesis — corrected executed evidence and reproducibility package',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'readiness_verdict': READINESS_VERDICT,
        'inventory_scope': INVENTORY_SCOPE,
        'payload_file_count': len(records),
        'archive_file_count_including_manifest_and_checksums': len(records) + 2,
        'payload_uncompressed_bytes': sum(record['bytes'] for record in records),
        'executed_notebook_count': 8,
        'final_docx_sha256': sha256_file(DOCX),
        'final_pdf_sha256': sha256_file(PDF),
        'final_manuscript_sha256': final['final_artifacts']['manuscript_markdown']['sha256'],
        'final_manuscript_whitespace_delimited_words':
            final['final_artifacts']['manuscript_markdown']['whitespace_delimited_words'],
        'execution_checks': '1648/1648 passed',
        'final_cross_artifact_checks': '15/15 passed',
        'benchmark_semantics': (
            'Every exported fold-specific training-mean forecast benchmark is '
            'independently checked against the horizon-specific supervised '
            'training-target interval.'
        ),
        'arima_correction': {
            'train_mean_rows_changed': 30240,
            'descriptive_financial_training_mean_lower_loss_cells': '1/16 (previously 4/16)',
            'forecasts_explanations_matched_primary_metrics_and_pairwise_tests_unchanged': True,
            'three_training_mean_dm_sign_changes_remain_nonsignificant': True,
        },
        'pdf_pages': final['pagination']['total_rendered_pages'],
        'main_text_physical_pages': final['pagination']['main_text_physical_pages'],
        'notebook_sha256': final['execution']['notebook_sha256'],
        'qa_scope': {
            'word_report_and_contact_sheet': 'qa/word_corrected_final/',
            'word_selected_full_resolution_pages': [4],
            'pdf_report_and_contact_sheet': 'qa/pdf_corrected_final/',
            'pdf_selected_full_resolution_pages': [4, 45, 47, 56, 86, 88, 89],
            'pdf_all_89_pages_inspected': True,
            'word_engine_pages': 84,
            'word_pages_rendered': 60,
        },
        'exclusions': [
            'audit/pre_arima_benchmark_correction/ and all defective superseded exports',
            'stale DOCX/PDF build intermediates and working copies',
            'old QA directories and unselected page-render files',
            '.DS_Store files',
            'stale package and external checksum artifacts',
        ],
        'file_records': records,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    sums_lines = [f"{record['sha256']}  {record['path']}" for record in records]
    sums_lines.append(f'{sha256_bytes(manifest_bytes)}  PACKAGE_MANIFEST.json')
    sums_bytes = ('\n'.join(sums_lines) + '\n').encode('utf-8')

    temporary = ARCHIVE.with_suffix(ARCHIVE.suffix + '.tmp')
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for record in records:
            archive.write(payload[record['path']], record['path'])
        archive.writestr('PACKAGE_MANIFEST.json', manifest_bytes)
        archive.writestr('SHA256SUMS.txt', sums_bytes)

    # Verify the new archive completely before atomically replacing the old one.
    # Renaming the verified file does not alter its bytes.
    zip_report = verify_zip(temporary, payload)
    os.replace(temporary, ARCHIVE)

    external_records = []
    for path in (DOCX, PDF, ARCHIVE, NOTES):
        external_records.append(f'{sha256_file(path)}  {path.name}')
    EXTERNAL_SUMS.write_text('\n'.join(external_records) + '\n', encoding='utf-8')
    external_verify = parse_sums(EXTERNAL_SUMS.read_text(encoding='utf-8'))
    for name, expected in external_verify.items():
        assert sha256_file(OUTPUT / name) == expected

    report = {
        'verified_utc': datetime.now(timezone.utc).isoformat(),
        'archive': str(ARCHIVE),
        'archive_sha256': sha256_file(ARCHIVE),
        'external_checksums': str(EXTERNAL_SUMS),
        'external_checksums_sha256': sha256_file(EXTERNAL_SUMS),
        'external_deliverable_checksums_verified': True,
        'readiness_verdict': READINESS_VERDICT,
        **zip_report,
        'all_passed': True,
    }
    VERIFICATION.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n',
                            encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
