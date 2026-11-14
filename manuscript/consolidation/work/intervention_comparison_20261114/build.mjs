import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../../../..');
const outputDir = path.join(root, 'outputs/intervention-comparison-20261114');
const previewDir = path.join(root, 'tmp/intervention-comparison-20261114');
const evidence = 'manuscript/consolidation/evidence/05-adam/c-intervention/';
const sources = {
  lr: evidence + 'observations/58-001-persistent-half-lr.md',
  cosine: 'analyses/044-2026-11-04-multispike-interventions/README.md',
  conventional: evidence + 'tables/conventional-outcomes.md',
  method: evidence + 'observations/65-015-conventional-model-coverage.md',
  cap: evidence + 'tables/distributed-cap-outcomes.md',
  capMethod: evidence + 'algorithm.md',
  ablation: evidence + 'observations/65-013-distributed-cap-quantile-lr.md',
  todo: evidence + 'TODO_ABLATIONS.md',
};
const sourceTexts = {};
const sourceHashes = {};
for (const [key, relative] of Object.entries(sources)) {
  const raw = await fs.readFile(path.join(root, relative));
  sourceTexts[key] = raw.toString('utf8');
  sourceHashes[relative] = crypto.createHash('sha256').update(raw).digest('hex');
}

// Source checks protect the principal numbers and known missing cells.
for (const value of ['-25.04%', '+27.20%', '-9.66%']) assert(sourceTexts.lr.includes(value));
for (const value of ['-26.30%', '+0.62%', '-10.68%']) assert(sourceTexts.cosine.includes(value));
for (const value of ['+16.78%', '+127.02%', '-3.53%', '-18.47%', '-11.82%', '+33.12%', '+2.08%', '+48.01%', '-28.32%', '+78.03%', '-19.02%', '-17.74%']) assert(sourceTexts.conventional.includes(value));
for (const value of ['-22.85', '-16.22', '-26.00', '-19.43', '-32.08', '+2.87', '+8.18', '+100.17']) assert(sourceTexts.cap.includes(value));
for (const value of ['54', '24.79%', '87.88%', '94.75%', '3.37%']) assert(sourceTexts.ablation.includes(value));
assert(sourceTexts.todo.includes('ResNet threshold .5'));

const columns = [
  'Intervention type',
  'Steps (setup)',
  'Parameters ablated',
  'Goal',
  'CIFAR-10 / MLP\nCoverage',
  'CIFAR-10 / ResNet-32\nCoverage',
  'WikiText-2 / Transformer\nCoverage',
  'Main conclusions',
  'Missing experiments',
  'Missing measurements / interpretation limits',
];
const rows = [
  [
    'Learning-rate schedules',
    'MLP: 512 to 800 (288 updates)\nResNet: 256 to 576 (320)\nTransformer: 544 to 928 (384)\n\nSame warm model and Adam state within each comparison.',
    'LR reduced to 0.5x:\nImmediate reduction\n32-update cosine transition\n\nBase LR: MLP 0.001; ResNet/Transformer 0.003.\nSelected PDF: immediate reduction.',
    'Test whether reducing the global step size suppresses loss excursions while preserving training progress.\n\nCompare abrupt versus gradual reduction.',
    'TESTED\nBoth schedules\nInitialization 0\n5,000 training images\nHalf squared output error',
    'TESTED\nBoth schedules\nInitialization 0\n50,000 training images\nFull-objective cross entropy',
    'TESTED\nBoth schedules\nInitialization 0\n2,088,600 training tokens\nFull-objective NLL',
    'Immediate halving: mean loss changes MLP -25.04%, ResNet +27.20%, Transformer -9.66%.\n\nCosine halving: -26.30%, +0.62%, -10.68%, respectively.\n\nImmediate halving removes detected >10% excursions only in the selected MLP window.',
    'Initializations 1 and 2 in all models.\nCommon forks and horizons with the other families.\nIndependent intervention times.\nBroader matched magnitude / transition-duration grid.',
    'Common layerwise cosine/RMS and output-motion histories.\nCommon held-out evaluation times and hardware cost.\n\nSelected fixed windows do not establish diagnostic-guided timing benefits.',
  ],
  [
    'Adam beta2 reductions',
    'MLP: 512 to 800 (288 updates)\nResNet: 256 to 576 (320)\nTransformer: 544 to 928 (384)\n\nChange beta2 once. Retain LR, moments and optimizer clock.',
    'beta2: 0.999 to 0.95, 0.9 or 0.5\n\nThree treatments per model.\nNine treatment/model combinations.\nNative bias correction uses the new beta2.',
    'Test whether faster second-moment adaptation changes excursion severity and training progress.\n\nThe tested schedule changes both adaptation and normalization.',
    'TESTED\n0.95 / 0.9 / 0.5\nInitialization 0\n5,000 training images',
    'TESTED\n0.95 / 0.9 / 0.5\nInitialization 0\n50,000 training images',
    'TESTED\n0.95 / 0.9 / 0.5\nInitialization 0\n2,088,600 training tokens',
    'MLP: every setting worsens mean loss (+16.78% to +127.02%).\nResNet: mixed; only 0.9 lowers mean (-3.53%), while all worsen late-window loss.\nTransformer: mean improves 11.82% to 18.47%; test loss worsens 24.64% to 33.12%.',
    'Initializations 1 and 2 in all models.\nCommon forks and horizons across families.\nIndependent times, duration and return to 0.999.\nBias-correction alternatives only if claiming an isolated memory effect.',
    'Layerwise representation histories were not captured in the model-coverage extension.\nCommon intermediate held-out times and cost.\n\nPreserved moments with new native bias correction do not isolate memory length.',
  ],
  [
    'Global gradient clipping',
    'MLP: 512 to 800 (288 updates)\nResNet: 256 to 576 (320)\nTransformer: 544 to 928 (384)\n\nClip the full raw gradient before Adam updates its moments.',
    'Global gradient L2 threshold:\nMLP: 0.5 and 1.0\nResNet: 1.0\nTransformer: 0.5 and 1.0\n\nFive treatment/model combinations.',
    'Test whether limiting optimizer-input gradient magnitude reduces excursions and improves optimization.\n\nKeep raw-gradient diagnostics separate from clipped inputs.',
    'TESTED\nThresholds 0.5 / 1.0\nInitialization 0\n5,000 training images',
    'PARTIAL\nThreshold 1.0 only\nInitialization 0\n50,000 training images\n\nMissing: threshold 0.5',
    'TESTED\nThresholds 0.5 / 1.0\nInitialization 0\n2,088,600 training tokens',
    'MLP: mean worsens +2.08% / +48.01% at thresholds 0.5 / 1.0.\nResNet: mean -28.32%, late-window mean +78.03%.\nTransformer: mean -19.02% / -17.74%, test +11.30% / +14.84% at 0.5 / 1.0.',
    'ResNet threshold 0.5.\nInitializations 1 and 2 in all models.\nCommon forks and horizons across families.\nIndependent times and a broader threshold-response grid.',
    'Common layerwise representation histories, intermediate held-out times and cost.\n\nClipping activity and optimizer-input norms are available. A raw-gradient cap does not directly bound the Adam increment.',
  ],
  [
    'Distributed displacement cap\nq50 main comparison',
    'MLP: 448 to 1152\nResNet: 264 to 576\nTransformer: 552 to 928\n\nBudgets use eight prior untreated updates.\nMLP heatmap displays 448 to 800.',
    'q50 budget per targeted module.\nScale proposed weight rows / biases after Adam.\n\nMLP: two hidden affines\nResNet: 31 convolutions\nTransformer: two QKV + four feed-forward affines',
    'Test whether controlling local representation movement damps harmful excursions while keeping feature learning active.\n\nDistinguish movement amplitude from reversal frequency.',
    'TESTED\nq50\nInitializations 0 / 1 / 2\n5,000 training images\nProbe: all 5,000 images',
    'TESTED\nq50\nInitialization 0\n50,000 training images\nProbe: intact 1,000-image group',
    'TESTED\nq50\nInitialization 0\n2,088,600 training tokens\nProbe: 64 sequences / 2,240 positions',
    'Seed-0 mean loss: MLP -22.85%, ResNet -16.22%, Transformer -26.00%. MLP seeds 1/2: -19.43% / -32.08%.\n\nReversals can persist or increase. Terminal test loss worsens in all five cases: seed-0 changes +2.87% / +8.18% / +100.17%.',
    'ResNet / Transformer initializations 1 and 2.\nCommon forks and horizons with conventional methods.\nIndependent intervention times.\nCalibration-window and probe-size sensitivity.',
    'Common cost and held-out times across families.\nCap means include the fork; conventional means begin at fork + 1.\n\nThe cap bounds the local affine response with inputs fixed, not total propagated motion or loss.',
  ],
  [
    'Distributed cap\nQuantile x LR ablation',
    'MLP: 448 to 1152\n705 measured loss states\n\nSame fork within initialization. Budgets calibrated at LR 0.001 and held fixed across LR settings.',
    'q25 / q50 / q75 / q90 / q99\nLR: 0.001 / 0.002 / 0.004 (1x / 2x / 4x)\nUncapped Adam at each LR\n\n18 conditions x 3 initializations = 54 continuations',
    'Measure sensitivity to cap strength.\nTest whether a larger nominal LR improves training under a fixed movement budget.\n\nCompare both with same-LR Adam and with the same cap at 1x LR.',
    'TESTED\nComplete 18-condition grid\nInitializations 0 / 1 / 2\n54 continuations\n5,000 training images',
    'FULL GRID MISSING\nNo matching quantile x LR x initialization sweep.\n\nSelected q50 comparison is available in the row above.',
    'FULL GRID MISSING\nNo matching quantile x LR x initialization sweep.\n\nSelected q50 comparison is available in the row above.',
    'q50 at 1x has the lowest observed mean loss (-24.79% paired across seeds).\nq25 lowers mean by 87.88% / 94.75% versus same-LR Adam at 2x / 4x.\nFor every quantile, higher LR worsens the across-seed mean versus its 1x setting. q50 at 1x raises final-four test mean 3.37%.',
    'ResNet and Transformer matching grids.\nIndependent forks.\nCalibration and probe sensitivity.\nAdditional MLP initializations if stronger uncertainty estimates are needed.',
    'Matched cross-architecture cost and common held-out times.\n\nMLP has 23 complete test evaluations per continuation. Final-four test mean differs from terminal test loss. Three seeds provide descriptive replication.',
  ],
];
assert.equal(rows.length, 5);
rows.forEach(row => assert.equal(row.length, columns.length));

const workbook = Workbook.create();
const sheet = workbook.worksheets.add('Intervention comparison');
sheet.showGridLines = false;
sheet.tabColor = '#203957';
sheet.getRange('A1:L10').format.font = { name: 'Arial', size: 11, color: '#202A35' };
sheet.getRange('A2').values = [['Intervention experiment comparison']];
sheet.getRange('A2').format.font = { name: 'Arial', size: 17, bold: true, color: '#203957' };
sheet.getRange('A3').values = [['Steps count completed optimizer updates. Loss changes compare each treatment with its own matched unchanged-Adam control.']];
sheet.getRange('A3').format.font = { name: 'Arial', size: 11, italic: true, color: '#55616F' };
sheet.getRange('A4').values = [['Four selected families, with the cap ablation shown separately. Family comparisons still require common forks, horizons and evaluation coverage.']];
sheet.getRange('A4').format.font = { name: 'Arial', size: 11, color: '#55616F' };
sheet.getRange('A5:J10').values = [columns, ...rows];
const table = sheet.tables.add('A5:J10', true, 'InterventionComparison');
table.style = 'TableStyleMedium2';
table.showFilterButton = true;
table.showTotals = false;

const widths = [190, 290, 300, 280, 225, 235, 250, 480, 370, 360];
for (let col = 0; col < widths.length; col++) {
  sheet.getRangeByIndexes(0, col, 10, 1).format.columnWidthPx = widths[col];
}
sheet.getRange('K1:K10').format.columnWidthPx = 30;
sheet.getRange('L1:L10').format.columnWidthPx = 540;
sheet.getRange('A1:L1').format.rowHeight = 12;
sheet.getRange('A2:L2').format.rowHeight = 28;
sheet.getRange('A3:L4').format.rowHeight = 23;
sheet.getRange('A5:J5').format = {
  fill: '#203957',
  font: { name: 'Arial', size: 11, bold: true, color: '#FFFFFF' },
  horizontalAlignment: 'center', verticalAlignment: 'center', wrapText: true,
  rowHeight: 44,
  borders: { insideVertical: { style: 'thin', color: '#FFFFFF' } },
};
sheet.getRange('A6:J10').format = {
  font: { name: 'Arial', size: 11, color: '#202A35' },
  horizontalAlignment: 'left', verticalAlignment: 'top', wrapText: true,
  rowHeight: 126,
  borders: { insideHorizontal: { style: 'thin', color: '#D4DBE4' } },
};
for (let row = 6; row <= 10; row++) {
  sheet.getRange(`A${row}:J${row}`).format.fill = row % 2 === 0 ? '#F2F5F9' : '#FFFFFF';
}
sheet.getRange('A6:A10').format.font = { name: 'Arial', size: 11, bold: true, color: '#203957' };
sheet.getRange('H6:H10').format.fill = '#EDF3FA';
sheet.getRange('I6:J10').format.fill = '#FFF3D6';
sheet.getRange('F8').format.fill = '#FCE4D6';
sheet.getRange('F10:G10').format.fill = '#FCE4D6';
sheet.freezePanes.freezeRows(5);
sheet.freezePanes.freezeColumns(1);

// Provenance remains beside the table, after a blank spacer column.
sheet.getRange('L5').values = [['Evidence sources']];
sheet.getRange('L5').format.font = { name: 'Arial', size: 11, bold: true, color: '#55616F' };
sheet.getRange('L6:L10').values = [
  ['Analysis 058, observation 001 (immediate half LR)\nAnalysis 044, completed schedule comparisons\nC-intervention ablation TODO'],
  ['Analysis 065, observation 015 and conventional outcomes\nAnalysis 059, observation 001 (native beta2 transition)\nC-intervention ablation TODO'],
  ['Analysis 065, observation 015 and conventional outcomes\nAnalysis 058, observation 005 (original clipping)\nC-intervention ablation TODO'],
  ['Distributed cap outcomes (five paired comparisons)\nAnalysis 065, observation 008 and Algorithm 2\nC-intervention ablation TODO'],
  ['Analysis 065, observation 013 (quantile x LR sweep)\nRuns 286, 287 and 288\nC-intervention ablation TODO'],
];
sheet.getRange('L6:L10').format = {
  font: { name: 'Arial', size: 11, color: '#55616F' },
  horizontalAlignment: 'left', verticalAlignment: 'top', wrapText: false,
};

workbook.recalculate();
assert.deepEqual(sheet.getRange('A5:J10').values, [columns, ...rows]);
const inspection = await workbook.inspect({
  kind: 'table', range: 'Intervention comparison!A5:J10', include: 'values,formulas',
  tableMaxRows: 6, tableMaxCols: 10, tableMaxCellChars: 90, maxChars: 5000,
});
const errors = await workbook.inspect({
  kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options: { useRegex: true, maxResults: 20 }, summary: 'final formula error scan', maxChars: 1500,
});
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
await fs.writeFile(path.join(previewDir, 'inspection.txt'), inspection.ndjson + '\n' + errors.ndjson);
for (const [label, range] of [['setup', 'A1:D10'], ['coverage-conclusions', 'E5:H10'], ['gaps-sources', 'I5:L10']]) {
  const preview = await workbook.render({ sheetName: sheet.name, range, scale: 1.4, format: 'png' });
  await fs.writeFile(path.join(previewDir, `${label}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const outputPath = path.join(outputDir, 'intervention-experiment-comparison.xlsx');
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);
const bytes = await fs.readFile(outputPath);
await fs.writeFile(path.join(here, 'verification.json'), JSON.stringify({
  workbook: path.relative(root, outputPath).replaceAll('\\', '/'),
  sha256: crypto.createHash('sha256').update(bytes).digest('hex'),
  bytes: bytes.length, worksheet: sheet.name, data_range: 'A5:J10',
  interventions: 5, comparison_columns: 10, exact_cell_readback: true,
  source_checks: 'passed', source_sha256: sourceHashes,
  visual_reviewed: false, saved_features_verified: false,
}, null, 2) + '\n');
console.log(JSON.stringify({ output: outputPath, rows: 5, columns: 10, bytes: bytes.length, error_scan: errors.ndjson }));
