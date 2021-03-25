def relative_to_dollar_conversion(rate_structure, basis):
    # Converts relative rate structure to dollar structure, to allow for calculation
    converted_structure = []
    for row in rate_structure:
        converted_structure.append([row[0] * basis, row[1] * basis, row[2]])

    return converted_structure


class Debt:
    def __init__(self, rate_structure=[[0, 1000, 0.02]], rate_structure_type='dollar', initial_debt=1000):
        assert rate_structure_type in ['dollar', 'relative'], 'Rate structure must be of type "dollar" or "relative"'
        self.debt_amount = initial_debt
        self.rate_structure_type = rate_structure_type
        self.rate_structure = rate_structure
        self.repaid = False

    def add_debt(self, debt_amount):
        self.debt_amount += debt_amount

    def prepayment(self, prepayment_amount):
        self.debt_amount -= min(prepayment_amount, self.debt_amount)
        if self.debt_amount <= 0:
            self.repaid = True

    def calculate_interest(self, basis=""):
        basis = self.debt_amount if basis == "" else basis
        interest_bill = 0
        rate_structure = relative_to_dollar_conversion(self.rate_structure, basis) if self.rate_structure_type != 'dollar' else self.rate_structure

        # Make sure entire debt is covered by rate structure
        rate_structure[-1][1] = self.debt_amount

        for row in rate_structure:
            basis -= row[0]
            interest_bill += min(row[1], basis) * row[2]

        return interest_bill



if __name__ == '__main__':
    SU = Debt(rate_structure=[[0, .4, 0.02], [.4, 1, 0.03]], rate_structure_type='relative', initial_debt=1000)
    print(SU.calculate_interest())
    print(SU.calculate_interest())
