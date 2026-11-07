// Tables 1-2 (Table 3, nested discrimination, withdrawn 20261031) for v65; originally Tables 1-3 for v54 as a landscape .docx with explicit column widths.
const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        WidthType, AlignmentType, PageOrientation, ShadingType, BorderStyle } = require('docx');

const ST = '20261030_v65';
const SUP = { '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵',
              '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻' };

function readCsv(p) {
  return fs.readFileSync(p, 'utf8').trim().split(/\r?\n/).map(line => {
    const out = []; let cur = '', q = false;
    for (const ch of line) {
      if (ch === '"') q = !q;
      else if (ch === ',' && !q) { out.push(cur); cur = ''; }
      else cur += ch;
    }
    out.push(cur); return out;
  });
}
function sci(v) {
  const n = Number(v);
  if (!isFinite(n) || v === '' || v === '—') return v;
  if (n >= 0.001) return String(Number(n.toPrecision(3)));
  const e = Math.floor(Math.log10(n));
  const m = (n / Math.pow(10, e)).toFixed(1);
  return `${m} × 10${String(e).split('').map(c => SUP[c]).join('')}`;
}
const minus = s => String(s).replace(/^-/, '−');

function cell(text, { bold = false, shade = false, width, align = AlignmentType.LEFT } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, fill: 'F2F4F6' } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ alignment: align,
      children: [new TextRun({ text: String(text), bold, size: 17, font: 'Calibri' })] })],
  });
}
function makeTable(rows, widths, aligns) {
  const border = { style: BorderStyle.SINGLE, size: 2, color: 'AAAAAA' };
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    borders: { top: border, bottom: border, left: border, right: border,
               insideHorizontal: border, insideVertical: border },
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0,
      children: r.map((c, j) => cell(c, { bold: i === 0, shade: i === 0,
                                          width: widths[j], align: aligns[j] })),
    })),
  });
}
const title = t => new Paragraph({ spacing: { before: 240, after: 120 },
  children: [new TextRun({ text: t, bold: true, size: 20, font: 'Calibri' })] });
const foot = t => new Paragraph({ spacing: { before: 120, after: 240 },
  children: [new TextRun({ text: t, italics: true, size: 16, font: 'Calibri' })] });

const NUM = {};
for (const r of readCsv(`numbers_v65_all_${ST}.csv`).slice(1)) NUM[r[0]] = r[1];
const ni = k => Math.round(Number(NUM[k])).toLocaleString('en-US');

// ---------------- Table 1 ----------------
const t1 = readCsv(`out/Table1_cohort_${ST}.csv`).map((r, i) =>
  r.map((c, j) => (i > 0 && j === r.length - 1) ? sci(c) : minus(c)));
const w1 = [2900, 1250, 1250, 1250, 1250, 1250, 1250, 1500];
const a1 = [AlignmentType.LEFT, ...Array(7).fill(AlignmentType.RIGHT)];

// ---------------- Table 2 ----------------
const raw2 = readCsv(`out/Table2_effects_${ST}.csv`);
const hdr2 = ['Exposure', 'Outcome', 'Analysis set', 'n', 'β (95% CI)', 'P', 'q'];
const body2 = raw2.slice(1).map(r => [r[0], r[1], r[2], r[3], r[4], sci(r[5]),
                                      r[6] === '' ? '—' : sci(r[6])]);
const w2 = [2400, 2800, 2100, 700, 2600, 1300, 1300];
const a2 = [AlignmentType.LEFT, AlignmentType.LEFT, AlignmentType.LEFT, AlignmentType.RIGHT,
            AlignmentType.RIGHT, AlignmentType.RIGHT, AlignmentType.RIGHT];

const doc = new Document({
  sections: [{
    properties: { page: { size: { orientation: PageOrientation.LANDSCAPE },
                          margin: { top: 720, bottom: 720, left: 720, right: 720 } } },
    children: [
      title(`Table 1. Characteristics of the profiled sample by CKM stage (n = ${ni('n_microbiome_all')})`),
      makeTable(t1, w1, a1),
      foot(`Values are mean (SD) for continuous variables and n (%) for binary variables. P values are Kruskal–Wallis across the five stages for continuous variables and chi-squared for binary variables. CKM stage is taken from the cohort's master data table at each participant's own blood-draw visit, which is the visit contemporaneous with stool collection (all within 180 days; median absolute interval ${NUM['mb_delta_days_median']} days); stages 0 to 2 are set by the cohort's health examination, stage 3 by cardiac, natriuretic-peptide, troponin or renal criteria met on or before that date, and stage 4 by a qualifying cardiovascular diagnosis dated before the draw in the linked hospital record. All clinical variables come from that same visit, and laboratory values, body-mass index, waist circumference and blood pressure may be carried forward from the most recent prior measurement within three years. Medication fields are populated only for visits at which diagnoses were extracted, so medication denominators are smaller than the column n (${ni('t1_med_denominator_overall')} overall) and medication percentages use those denominators. BP, blood pressure; eGFR, estimated glomerular filtration rate; HDL, high-density lipoprotein; SGLT2, sodium–glucose cotransporter 2.`),

      title('Table 2. The pre-specified three-genus score and the cross-validated panel against the ether-lipid readouts and against clinical outcome'),
      makeTable([hdr2, ...body2], w2, a2),
      foot(`β is expressed in standard deviations of the outcome per standard deviation of the exposure, except the two log-odds rows. "Metabolomics + 16S" is the ${ni('n_microbiome')} participants with both layers, in which any metabolite outcome is estimable; "All 16S profiled" is all ${ni('n_microbiome_all')} participants with 16S and a matched clinical visit. q values are Benjamini–Hochberg across the clinical-outcome family; the metabolite rows are not members of that family and are marked with an em dash. The pre-specified three-genus score is the primary exposure; the cross-validated panel is scored out of fold and is reported as a secondary, data-driven exposure. All models adjust for age, sex, body-mass index and sequencing batch, with HC3 standard errors.`),

    ],
  }],
});
Packer.toBuffer(doc).then(b => {
  fs.writeFileSync(`out/Tables1-2_${ST}.docx`, b);
  console.log(`Tables1-2_${ST}.docx written (${t1.length - 1} + ${body2.length} data rows)`);
});
