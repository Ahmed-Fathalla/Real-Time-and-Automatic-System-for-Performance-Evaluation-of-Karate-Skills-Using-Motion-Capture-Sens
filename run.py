from utils.skill import *

skill_dict = {
                1:'GEDAN BARAI',
                2:'OI ZUKI',
                3:'SOTO UKE',
                4:'AGE UKE',
            }

for skill in range(1,5):
    create_model(skill, draw = True)