''''    |     cancer    | Total
symptom |No     | Yes   |
 no     |99989  | 0     | 99989
 yes    |10     | 1     | 11
 total  |99999  | 1     | 100000
'''


def cal_bayes(prior_H, prob_E_given_H, prob_E):
    return prior_H * prob_E_given_H / prob_E


if __name__ == '__main__':
    prior_H = 1 / 100000  # probability of hypothesis (has cancer), some times is not better than a guess,
    # will be updated as new information comes in
    prob_not_H = 1 - prior_H

    prob_E_given_H = 1  # all people with cancer have symptoms 100% (according to this data)
    prob_E_given_not_H = 10 / 99999  # 10 out of 99999 have symptoms but not cancer

    prob_E = prior_H * prob_E_given_H + prob_not_H * prob_E_given_not_H # all the cases of symptoms

    print("The probability of having cancer given symptoms or P(H|E) is: \n", cal_bayes(prior_H, prob_E_given_H, prob_E))

'''
H / hypothesis = has cancer = 1 / 100 000
E / Event = has symptoms

P(E|H) = has symptoms given cancer; the data tells us that all the people with cancer has symptoms 
Therefore, the probability of having symptoms if you already have cancer is 1 or 100%
P(E|H) = 1

P(E|not H) = has symptoms given not cancer;  the data tells us that 10 people out of 99999 have symptoms but not cancer 
P(E|not H) = 10 / 99999

Therefore, The probability of having cancer given symptoms is: 

P(H|E) = P(H) * P(E|H) / P(E) 

where P(E) is the sum of all the cases of symptoms

P(E) = P(H) * P(E|H) + P(no H) * P(E|no H)


***************************** AND THIS IS BAYES THEOREM *************************
******    P(H|E) = P(H) * P(E|H) / (P(H) * P(E|H) + P(no H) * P(E|no H))    *****



another point of view:
in the area H is 1 person with symptoms, the total of people is 1
in the area (no H) are 10 people with symptoms, the total of people (no H) is 99999

so the probability of have cancer if you have symptoms is 1 / 11
even more
the people with cancer that have symptoms divided by the total of people with symptoms
people with cancer and symptoms = P(H)*P(E|H)
people with symptoms and not cancer = P(no H)*P(E|no H)
P(H|E) = P(H)*P(E|H) / P(H)*P(E|H) + P(no H)*P(E|no H)
             H     no H
            -------------
            |   |       |
            |   |       |   
            |   |       |
            |   |       |
            |   |       |  
            ------------- 
'''
