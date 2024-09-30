import requests


class CurrencyConverter:

    def __init__(self, local_currency: str, foreign_currency: str):
        self.local_currency = local_currency.upper()
        self.foreign_currency = foreign_currency.upper()

        self.currency_exchange()

    def currency_exchange(self):

        url = f"https://api.exchangerate-api.com/v4/latest/{self.local_currency}"
        try:
            response = requests.get(url)
            data = response.json()
            rates = data['rates']

            # Get desired rate:
            self.rate = rates[self.foreign_currency]
            print(f"1 {self.local_currency} = {self.rate} {self.foreign_currency}")

        except Exception as e:
            print(f"Error: {e}")

    def get_exchange_rate(self):
        return self.rate

    def amount_in_local_currency(self, foreign_amount=1):
        return foreign_amount / self.rate

    def amount_in_foreign_currency(self, local_amount=1):
        return local_amount * self.rate