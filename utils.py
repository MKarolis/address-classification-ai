import pandas as pd

def read_DataFrame_from_file(filename: str, numberOfRows: int = None):
    return pd.read_excel(filename, nrows = numberOfRows, keep_default_na=False)


def write_DataFrame_to_excel(df: pd.DataFrame, filename: str):
    sheet_name = 'Output'

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        worksheet = writer.sheets[sheet_name]
        # format all data as a table
        worksheet.add_table(0, 0, df.shape[0], df.shape[1]-1, {
            'columns': [{'header': col_name} for col_name in df.columns],
            'style': 'Table Style Medium 5'
        })
        # Widen the address column
        worksheet.set_column('A:A', 70)


def get_precision_recall_f1(actual: list, predicted: list):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for index, predictedVal in enumerate(predicted):
        actualVal = actual[index]
        if (actualVal and actualVal == predictedVal):
            true_positives += 1
        elif (predictedVal and actualVal != predictedVal):
            false_positives += 1
        elif (actualVal and actualVal != predictedVal):
            false_negatives += 1
    
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * ((precision * recall) / max(precision + recall, 1))

    return [el * 100 for el in [precision, recall, f1]]