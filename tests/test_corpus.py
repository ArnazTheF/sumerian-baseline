"""Проверки парсинга, идентификаторов и разбиений корпуса."""
import json
import tempfile
import unittest
from pathlib import Path

from sumbase.corpus.parsers import (atf_documents, cdl_lines, etcsl_xml,
                                    fingerprint, oracc_html_segments)
from sumbase.corpus.quality import screen_mt_near_duplicates
from sumbase.corpus.media import maicubeda_member
from sumbase.corpus.bdtns_http import BDTNSPublicSession


class CorpusTests(unittest.TestCase):
    def test_translit_lines_without_source_ids_remain_distinct(self):
        import sqlite3
        from sumbase.corpus.build import export_translit, pack
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'exports').mkdir()
            db = sqlite3.connect(':memory:')
            db.executescript('CREATE TABLE editions(uid,source,text_id,group_id,data); CREATE TABLE excluded_documents(uid);')
            lines = [{'id':'','glyphs':glyph,'text':text,'is_sumerian':True}
                     for glyph,text in [('𒀀','a'),('𒁀','ba'),('𒂍','e2')]]
            db.execute('INSERT INTO editions VALUES(?,?,?,?,?)',
                       ('example:tablet','example','P000001','group1',pack({'lines':lines})))
            export_translit(root,db)
            path = root/'exports/translit/external_candidates.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows),3)
            self.assertEqual(len({r['id'] for r in rows}),3)
            before = path.read_bytes()
            export_translit(root,db)
            self.assertEqual(path.read_bytes(),before)
            db.close()

    def test_bdtns_reverse_first_line_and_determinative_are_preserved(self):
        from sumbase.corpus.parsers import bdtns_documents
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'records').mkdir()
            item = {'bdtns_id':'000001','cdli_ids':['P000001'],'catalogue_text':'example','query_record_url':'https://bdtns.cesga.es/example'}
            (folder/'novel-plan.json').write_text(json.dumps({'records':[item]}))
            (folder/'records/000001.html').write_text('<div>BDTNS 000001 =</div><table><tr><td width="10%">1</td><td width="90%">2 udu</td></tr><tr><td width="10%">r. 1</td><td width="90%">mu <sup>d</sup>nanna</td></tr></table>')
            row = next(bdtns_documents(folder))
            self.assertEqual([x['text'] for x in row['lines']], ['2 udu','mu {d}nanna'])
    def test_mt_adapter_preserves_original_and_converts_cdli_notation(self):
        from sumbase.config import Config
        from sumbase.data.prepare_collection import prepare_collection
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'exports/mt'
            folder.mkdir(parents=True)
            row = {'id':'line1','src':'ša₃ e₂ ĝeš','tgt':'inside the temple'}
            for split in ('train','valid','external_candidates'):
                (folder/f'{split}.jsonl').write_text(json.dumps(row)+'\n')
            cfg = Config(task='mt',data={'collection_root':str(root),'test_source':'external_candidates','input_normalization':'cdli_ascii'},paths={'data_dir':str(root/'ready')})
            prepare_collection(cfg)
            prepared = json.loads((root/'ready/test.jsonl').read_text())
            self.assertEqual(prepared['src'],'sza3 e2 jesz')
            self.assertEqual(prepared['src_original'],row['src'])
            self.assertEqual(json.loads((folder/'external_candidates.jsonl').read_text())['src'],row['src'])
    def test_bdtns_session_error_is_not_a_record(self):
        self.assertFalse(BDTNSPublicSession.matches(b'<html>Error en la busqueda de los datos</html>', '000123'))
        self.assertFalse(BDTNSPublicSession.matches(b'<td>BDTNS 000124 =</td>', '000123'))
        self.assertTrue(BDTNSPublicSession.matches(b'<td>BDTNS 000123 =</td>', '123'))

    def test_bdtns_page_offset_and_current_reference(self):
        content = b'''<html>Texts from 101 to 200 out of 94597 found<table><tr>
            <td><a href="ficha_simple_ventana.php?miReferencia=3">Text</a></td>
            <td><a href="resultado_bibliografia.php?id_bdts=000123">Bibliography</a></td>
            </tr></table></html>'''
        self.assertEqual(BDTNSPublicSession.parse_page(content, 101), {123:'ficha_simple_ventana.php?miReferencia=3'})
        with self.assertRaises(ValueError):
            BDTNSPublicSession.parse_page(content, 201)

    def test_end_to_end_collection_preserves_but_excludes_modern_fake(self):
        from sumbase.corpus.build import build, connect, unpack
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root/'raw/cdli'
            raw.mkdir(parents=True)
            (raw/'cdli_cat.csv').write_text('id_text,genre,language\n1,Administrative,Sumerian\n2,fake (modern),Sumerian\n')
            (raw/'cdliatf_unblocked.atf').write_text('&P000001 = tablet\n#atf: lang sux\n@obverse\n1. 2(disz) udu\n#tr.en: two sheep\n&P000002 = forgery\n#atf: lang sux\n@obverse\n1. lugal e2 mu-du3\n#tr.en: the king built the temple\n')
            build(root)
            db = connect(root)
            fake = unpack(db.execute("SELECT data FROM editions WHERE text_id='P000002'").fetchone()[0])
            self.assertIn('catalogued_modern_fake', fake['quality_flags'])
            self.assertEqual(db.execute('SELECT count(*) FROM segments').fetchone()[0],2)
            db.close()
            examples = []
            for split in ('train','valid','test','external_candidates'):
                examples.extend(json.loads(line) for line in (root/f'exports/mt/{split}.jsonl').read_text().splitlines())
            self.assertEqual(len(examples),1)
            self.assertEqual(examples[0]['tgt'],'two sheep')

    def test_maicubeda_published_filename_mismatch(self):
        self.assertEqual(maicubeda_member("1_3(disz)_1_1_1_2_HS_1137_06_back.jpg"),
                         "char/1_3(disz)_1_1_2_HS_1137_06_back.png")
    def test_atf_reused_line_numbers_and_language_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.atf"
            path.write_text("&P000001 = one\n#atf: lang sux\n@obverse\n1. 2(disz) udu\n#tr.en: two sheep\n@reverse\n1. mu lugal\n#tr.en: king's year\n&P000002 = two\n#atf: lang akk\n1. LUGAL\n#tr.en: king\n")
            rows = list(atf_documents(path, "test", "CC0"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]["segments"]), 2)
            self.assertNotEqual(rows[0]["segments"][0]["id"], rows[0]["segments"][1]["id"])

    def test_oracc_many_lines_to_one_translation(self):
        content = b'''<table class="transliteration"><tr id="P000001.1" class="l"><td class="xtr" id="P000001_project-en.0" rowspan="2"><span class="xtr-label">(1)</span>For Enlil, king of all the lands, his lord.</td></tr><tr id="P000001.2" class="l"/></table>'''
        lines = [{"id": "P000001.1", "text": "{d}en-lil2", "is_sumerian": True},
                 {"id": "P000001.2", "text": "lugal kur-kur-ra", "is_sumerian": True}]
        rows = oracc_html_segments(content, lines)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["src"], "{d}en-lil2\nlugal kur-kur-ra")
        self.assertNotIn("(1)", rows[0]["tgt"])
        lines[1]["is_sumerian"] = False
        self.assertEqual(oracc_html_segments(content, lines), [])

    def test_etcsri_style_missing_text_row_rejected(self):
        content = b'<table><tr id="Q000001.1" class="l"><td class="xtr" id="Q000001_project-en.0" rowspan="2">The king built the temple.</td></tr><tr id="Q000001.2" class="l"/></table>'
        lines = [{"id":"Q000001.1","text":"lugal","is_sumerian":True}]
        self.assertEqual(oracc_html_segments(content, lines), [])

    def test_german_with_technical_en_key_is_detected(self):
        content = '<table><tr id="Q000001.1" class="l"><td class="xtr" id="Q000001_project-en.0">In jenen alten Tagen, als gute Schicksale entschieden wurden, erschufen die Götter die Länder und die Menschen.</td></tr></table>'.encode()
        lines = [{"id":"Q000001.1","text":"u4 re-a","is_sumerian":True}]
        rows = oracc_html_segments(content, lines)
        self.assertEqual(rows[0]["target_lang"], "de")

    def test_notation_normalization_keeps_numbers_and_damage(self):
        self.assertEqual(fingerprint("ša₃ ĝeš₂"), fingerprint("sza3 jesz2"))
        self.assertNotEqual(fingerprint("2 udu"), fingerprint("3 udu"))
        self.assertNotEqual(fingerprint("[lugal]"), fingerprint("lugal"))

    def test_etcsl_entities_no_network_no_lost_determinatives(self):
        tree, unknown = etcsl_xml(b'<TEI.2><w form="&d;en-ki &ki; &mucen;">&h; &c;</w></TEI.2>')
        self.assertEqual(tree[0].get("form"), "{d}en-ki {ki} {mušen}")
        self.assertFalse(unknown)

    def test_near_duplicate_long_passage_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            text = "a ba c da e f g h i j k l m n o p q r s t u v"
            (folder/"train.jsonl").write_text(json.dumps({"id":"train1","src":text,"tgt":"example"})+'\n')
            (folder/"test.jsonl").write_text(json.dumps({"id":"test1","src":text+' zz',"tgt":"different reference"})+'\n')
            for split in ("valid", "external_candidates"):
                (folder/f"{split}.jsonl").write_text("")
            counts = screen_mt_near_duplicates(folder)
            self.assertEqual(counts["test"], 1)
            self.assertEqual((folder/"test.jsonl").read_text(), "")


if __name__ == "__main__":
    unittest.main()
