# This file was to merge disaster declarations summaries(col: declarationType, declarationTitle) into ihpvr with disaster number as a key.
import pandas as pd

ihp_vr_path = "E:\\CIS590\\15.FEMA\\IndividualAssistance\\IndividualsAndHouseholdsProgramValidRegistrations.csv" #E:\CIS590\15.FEMA\IndividualAssistance
declarations_path = "E:\\CIS590\\15.FEMA\\DisasterDeclarations\\DisasterDeclarationsSummaries.csv"
output_path = "E:\\CIS590\\15.FEMA\\Merged_IHP_VR.csv"

dtype_dict = {
    "disasterNumber": "int32",
    "declarationType": "category",
    "declarationTitle": "category"
}

IHP_VR = pd.read_csv(ihp_vr_path, usecols=["disasterNumber"], dtype={"disasterNumber": "int32"}, low_memory=True)
Declarations = pd.read_csv(declarations_path, usecols=["disasterNumber", "declarationType", "declarationTitle"], dtype=dtype_dict)

merged_data = pd.merge(IHP_VR, Declarations, on="disasterNumber", how="left")

chunk_size = 100000
merged_data.to_csv(output_path, index=False, chunksize=chunk_size)

print("The new dataset saved to:", output_path)
