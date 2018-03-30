# FROC: Free-response Receiver Operating Characteristic curve

This script computes and plots a FROC - Free-response Receiver Operating Characteristic. <br />
FROC curve is an alternative to ROC curve. On the x-axis stands the average number of false positives (FP) per scan instead of the false positive rate (FP/N, with N number of negatives). This plot is particularly useful for unbalanced detection problems, where the number of positives P is significantly lower than N. This would result in sticking all the meaningful information to the left of a ROC curve plot, and make any interpretation difficult.

Examples of FROC curves:<br />
https://www.researchgate.net/publication/304163398_Deep_Learning_for_Identifying_Metastatic_Breast_Cancer/figures
Figure 5: https://www.researchgate.net/publication/51369054_Pulmonary_Nodules_on_Multi-Detector_Row_CT_Scans_Performance_Comparison_of_Radiologists_and_Computer-aided_Detection1/figures?lo=1

References:<br />
[1] Bandos, A.I., Rockette, H.E., Song, T. and Gur, D., 2009. Area under the Free‐Response ROC Curve (FROC) and a Related Summary Index. Biometrics, 65(1), pp.247-256.
[2] https://en.wikipedia.org/wiki/Receiver_operating_characteristic
