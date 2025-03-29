import pandas as pd

input_file = "../dataset/twitter_dataset.csv"
output_file = "../dataset/cleaned_twitter_dataset.csv"


################# HELPER FUNCTIONS #################


def clean_text(text):
    if pd.isna(text):
        return ""

    # Remove newlines and extra spaces, keep text within single quotes
    return " ".join(str(text).replace("\n", " ").replace("\r", " ").split())


def clean_data(df):

    df_cleaned = df.copy()

    # Clean and process the data
    df_cleaned.columns = ["id", "entity", "sentiment", "text"]

    # Apply cleaning to text column
    df_cleaned["text"] = df_cleaned["text"].apply(clean_text)

    # Sort by id
    df_cleaned = df_cleaned.sort_values("id")

    return df_cleaned


################# MAIN FUNCTION #################


def data_ingestion(input_file):
    """
    Ingests the raw dataset from the specified input file.
    Args:
        input_file (str): Path to the input CSV file.
    Returns:
        pd.DataFrame: DataFrame containing the raw dataset.
    """

    df = pd.read_csv(
        input_file,
        quotechar='"',
        on_bad_lines="skip",
        encoding="utf-8",
    )

    return df


def data_preprocessing(df):

    # Clean and process the data
    df_cleaned = clean_data(df)

    # Save the cleaned dataset
    df_cleaned.to_csv(
        output_file, index=False, quoting=1
    )  # QUOTE_ALL to preserve quotes


if __name__ == "__main__":
    # Ingest the raw dataset
    df = data_ingestion(input_file)

    # Preprocess the data
    data_preprocessing(df)
