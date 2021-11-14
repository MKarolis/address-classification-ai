import pandas as pd
import re
from utils import read_DataFrame_from_file, write_DataFrame_to_excel

DATA_INPUT_FILENAME = 'raw_data.xlsx'
DATA_OUTPUT_FILENAME = 'training_data.xlsx'
NUMBER_OF_PARSABLE_RECORDS = 999

PROPERTY_LABEL = 'label'
PROPERTY_ADDRESS = 'address'

PROPERTY_LENGTH = 'length'
PROPERTY_DIGITS_COUNT = 'digit_count'
PROPERTY_DIGITS_GROUP_COUNT = 'digits_group_count'
PROPERTY_SEPARATED_DIGIT_GROUP_COUNT = 'separated_digits_group_count'
PROPERTY_TOKEN_COUNT = 'token_count'
PROPERTY_COMMA_COUNT = 'comma_count'
PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS = 'comma_separated_entities_having_numbers'
PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS_NEAR_WORDS = 'comma_separated_entities_having_numbers_near_words'


# Gets the length of address
def getAddressLength(address: str):
    """

    Gets the length of address.

    Args:
        address (str): address

    Returns:
        length: of the address

    """
    return len(address)


# Counts digits within an address
def getDigitsCount(address: str):
    """

    Counts digits within an address.

    Args:
        address (str): address

    Returns:
        digitsCount: within an address

    """
    return len(re.findall(r'\d', address))


# Count how many groups of consecutive digits are there
def getDigitGroupCount(address: str):
    """

    Count how many groups of consecutive digits are there.

    Args:
        address (str): address

    Returns:
        digitGroupCount: within an address

    """
    return len(re.findall(r'(\d+)', address))


# Count how many separated groups of consecutive numbers are there, for instance, (1-123, 124 565, 12/27)
def getSeparatedDigitGroupCount(address: str):
    """

    Count how many separated groups of consecutive numbers are there, for instance, (1-123, 124 565, 12/27).

    Args:
        address (str): address

    Returns:
        separatedDigitGroupCount: within an address

    """
    return len(re.findall(r'(\d+)[^\d,](\d+)', address))


# Count the number of tokens, separated by spaces or commas
def getTokenCount(address: str):
    """

    Count the number of tokens, separated by spaces or commas.

    Args:
        address (str): address

    Returns:
        tokenCount: within an address

    """
    return len(re.findall(r'([^\s,]+)', address))


# Counts the number of commas
def getCommaCount(address: str):
    """

    Counts the number of commas.

    Args:
        address (str): address

    Returns:
        commaCount: within an address

    """
    return len(re.findall(r',', address))


# Counts number of comma separated entities, having a number within them
def getCommaSeparatedEntityWithNumbersCount(address: str):
    """

    Counts number of comma separated entities, having a number within them.

    Args:
        address (str): address

    Returns:
        commaSeparatedEntityWithNumbersCount: within an address

    """
    return len(re.findall(r'[^,]*\d+[^,]*', address))


# Counts number of comma separated entities, having a number and a separated word:
def getCommaSeparatedEntityWithNumbersNearWordsCount(address: str):
    """

    Counts number of comma separated entities, having a number and a separated word.

    Args:
        address (str): address

    Returns:
        commaSeparatedEntityWithNumbersCount: within an address

    """
    return len(re.findall(r'([a-zA-Z]{3,})\s(\w+-)?\d+(-\w+)?|(\w+-)?\d+(-\w+)?\s([a-zA-Z]{3,})[^,]*', address))


def enrichDataFrameWithProperties(frame: pd.DataFrame):
    """

    Enrichs a dataframe with derivate features like length, token count, etc...

    Args:
        frame (pd.DataFrame): dataframe to be enrichen

    """
    assignProperty(frame, PROPERTY_LENGTH, getAddressLength)
    assignProperty(frame, PROPERTY_DIGITS_COUNT, getDigitsCount)
    assignProperty(frame, PROPERTY_DIGITS_GROUP_COUNT, getDigitGroupCount)
    assignProperty(frame, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT, getSeparatedDigitGroupCount)
    assignProperty(frame, PROPERTY_TOKEN_COUNT, getTokenCount)
    assignProperty(frame, PROPERTY_COMMA_COUNT, getCommaCount)
    assignProperty(frame, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS, getCommaSeparatedEntityWithNumbersCount)
    assignProperty(frame, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS_NEAR_WORDS, getCommaSeparatedEntityWithNumbersNearWordsCount)


def assignProperty(dataFrame: pd.DataFrame, property: str, fun: callable):
    """

    Derivate a feature and assign it to a dataframe as a column

    Args:
        dataFrame (pd.DataFrame): dataframe to be assign a feature
        property (str) : feature to assign to the dataframe
        fun (callable): function responsible to derivate the feature

    """
    dataFrame[property] = dataFrame.apply(lambda row: fun(row[PROPERTY_ADDRESS]), axis=1)


if __name__ == '__main__':
    
    rawData = read_DataFrame_from_file(DATA_INPUT_FILENAME, NUMBER_OF_PARSABLE_RECORDS)

    parsedData = pd.DataFrame()
    parsedData[PROPERTY_ADDRESS] = rawData['person_address']
    enrichDataFrameWithProperties(parsedData)
    parsedData[PROPERTY_LABEL] = rawData['label']

    write_DataFrame_to_excel(parsedData, DATA_OUTPUT_FILENAME)