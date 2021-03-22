import pandas as pd
import numpy as np
import arch
import datetime as dt
import scipy.stats as stats


class Debt:
    def __init__(self, rate_structure = [[0, , 0.02]], rate_structure_type = 'dollar'):
        self.debt_amount = 0
        self.rate_structure = rate_structure
        self.rate_structure_type = rate_structure_type
        self.repaid = False
    
    
    def change_rate_structure(self, rate_structure):
        self.rate_structure = rate_structure
        
    
    def add_debt(self, debt_amount):
        self.debt_amount += debt_amount
        
        
    def prepayment(self, prepayment_amount):
        self.debt_amount -= prepayment_amount
        if self.debt_amount <= 0:
            self.repaid = True
        
        
    def calculate_interest(self, basis = self.debt_amount):
        interest_bill = 0
        if rate_structure_type = 'dollar':
            for row in self.rate_structure:
                basis -= row[0]
                interest_bill += min(row[1], basis)*row[2]
            
            return interest_bill
        
        else:
            debt = self.debt_amount
            for row in self.rate_structure:
                basis *= (row[0]-row[1])
                interest_bill += 
                debt -= basis
            return interest_bill