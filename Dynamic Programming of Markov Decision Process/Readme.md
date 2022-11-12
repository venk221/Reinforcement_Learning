- Note: Our environment code mdp_dp.py and mdp_dp_test.py were developed for gym of version 0.25.2. 

- OpenAI updated their gym to the latest version of 0.26. 

- So when you install your gym, please use command line "pip install gym==0.25.2". 

- If you have installed other version of gym, "pip install gym==0.25.2" makes sure you will get 0.25.2.  

- If you want to check which version of gym you have, please use the code as below:

from gym.version import VERSION
print(VERSION)

Evaluate functions by typing "nosetests -v mdp_dp_test.py" in terminal (you need put mdp_dp.py and mdp_dp_test.py in the same folder)
If you got error using "nosetests -v mdp_dp_test.py" due to python version (sometimes, nosetests will use python2.7 by default), try: python3 -m nose -v mdp_dp_test.py
