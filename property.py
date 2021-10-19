import torch
import torch.nn as nn
from torch import optim
import torch.functional as F
import pytorch_lightning as pl

from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split



class Money1:
    
    def __init__(self, dollars, cents):
        self.dollars = dollars
        self.cents = cents
        
    
class Money:
    def __init__(self, dollars, cents):
        self.total_cents = dollars * 100 + cents
        
    @property
    def dollars(self):
        return self.total_cents / 100

    @dollars.setter
    def dollars(self, new_dollors):
        self.total_cents = 100 * new_dollors + self.cents

    # And the getter and setter for cents.
    @property
    def cents(self):
        return self.total_cents % 100

    @cents.setter
    def cents(self, new_cents):
        self.total_cents = 100 * self.dollars + new_cents

    def add(*argv):
        return sum(i for i in argv)
    
    def add2(yourMoney):
        print(yourMoney)

if __name__ == "__main__":
    coins = Money(10, 120)
    print(coins.dollars)
    coins.dollars = 500
    print(coins.dollars)
    
