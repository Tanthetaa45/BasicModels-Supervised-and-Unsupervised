import random
import math
from math import ceil, floor

def roll_dice():
    die1=random.randrange(1,7)
    die2=random.randrange(1,7)
    return (die1,die2)
def display_dice(dice):
    die1,die2=dice
    print(f'player rolled {die1}+{die2}={sum(dice)}')

die_values=roll_dice()
display_dice(die_values)
sum_of_dice=sum(die_values)

if sum_of_dice in (7,11):
    game_status='WON'
elif sum_of_dice in (2,3,12):
    game_status='LOST'
else:
    game_status='CONTINUE'
    my_point=sum_of_dice
    print('Point is',my_point)

while game_status=='CONTINUE':
    die_values=roll_dice()
    display_dice(die_values)
    sum_of_dice=sum(die_values)

    if sum_of_dice==my_point:
        game_status='WON'
    elif sum_of_dice==7:
        game_status='LOST'

if game_status=='WON':
    print('Player wins')
else:
    print('Player Loses') 
# *args makes the function receieve any number of additional arguments and pack them into a tuple

x=7
def access_global():
    print('x printed from access_global:',x)

access_global()

width=15.5
print('id:',id(width),'value:',width)
