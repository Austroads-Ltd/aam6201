CONFIG = dict(
    njobs=6,
    objective_col_with_treatment='dnPCI',
    objective_col_without_treatment='dnDNPCI',
    cost='Cost',
    metro_penalty_col='Metro',
    freight_penalty_col='Freight',
    los_after_with_treatment='nPCI_After',
    los_after_without_treatment='nDNPCI_After',
    committed_treatment_col="com_trt",
    budget=2*(10**8),
    len_col="insp_length",
    lane_col="lanes",
    lane_len_col="lane_length",
    output_filename='aggregated_results.csv',
    input_filename='NLTP_Unlimited_dTAGTL.csv'
)
