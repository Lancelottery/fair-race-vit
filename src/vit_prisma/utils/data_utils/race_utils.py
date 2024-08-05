
from vit_prisma.utils.data_utils.race_dict import RACE_DICT

def race_index_from_word(search_term):
    # Convert the search term to lowercase to ensure case-insensitive matching
    search_term = search_term.upper()

    # Iterate over the dictionary and search for the term
    for key, value in RACE_DICT.items():
        if search_term in value.upper():  # Convert each value to lowercase for case-insensitive comparison
            return key  # Return the key directly once found

    # If the loop completes without returning, the term was not found; raise an exception
    raise ValueError(f"'{search_term}' not found in RACE_DICT.")