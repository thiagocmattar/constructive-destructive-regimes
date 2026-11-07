# Original single-rate figure snapshot

Preserves Analysis060 before the requested LR0.04/0.08 extension. Sources and
generator are frozen at commit cc26dac; the copied generator is provenance,
not executable from this archive path (its relative paths assume Analysis060).
Restore it to the analysis root in an isolated checkout to reproduce the old
figure. Observation001 describes this snapshot. Current 01_report.py generates
the three-column figure and checks that these retained values are unchanged.
