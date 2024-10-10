from iso3166 import countries

def get_country_codes():
    return {
        country.alpha2: country.name for country in countries
    }