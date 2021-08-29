# blackjack_nn
Perceptron-based neural network learns to play blackjack


What is it?
-----------

This project is an experiment showing how a simplу neural network 
can automate the process of gambling (blackjack).

The program is written on python 3.6.9 in Google Colaboratory

It has code comments on RU

Rules
-----

Neural network plays blackjack by American rules:
   - dealer has 1 card open, 1 card closed
   - if dealer have 17, he stand.

I took them from wiki: https://en.wikipedia.org/wiki/Blackjack

But for the sake of simplicity, I left only actions
   - hit
   - stand

In a future, I will add other decisions, such as
   - double down
   - split
   - surrender


How it work? / Code structure
-----------------------------

1. Import libraries
   - numpy
   - tensorflow
   - matplotlib.pyplot
 
2. Generate data for machine learning 
   - def get_cards(): # deal cards for player and dealer
   - def get_data_from_cards(): # gives expected from nn output 

3. Building and training a neural network
~~~ python
model = Sequential([
          InputLayer((2,)),
          Dense(32, activation='relu'),
          Dense(1, activation='tanh')
])
~~~
~~~ python
model.compile(
    optimizer=Adam(0.001),
    loss='mean_squared_error',
    metrics=['accuracy']
)
~~~
~~~ python
model.fit(X, Y, batch_size=32, epochs=25, verbose=0)
~~~
4. Using nn in games with auto dealer and virtual money balance

![bal_plot](https://user-images.githubusercontent.com/80642434/131209304-e8282280-80e2-4ba1-90e2-616bc2696f9c.png)
*balance plot after 500 games*

How to try
----------

Google Colaboratory:
   - copy from 
   https://colab.research.google.com/drive/12gk_r1jcvCL96CiPstF61wct804AudiT?usp=sharing

Jupyter notebook:
   - download .ipynb file
   - write this in a start of program:
~~~
!pip install numpy
!pip install tensorflow
!pip install matplotlib
~~~

Windows/MacOS/Linux and other platforms:
   - download .py file
   - https://numpy.org/install/
   - https://www.tensorflow.org/install
   - https://matplotlib.org/stable/users/installing.html


Authors
-------

- spaceshine
https://github.com/spaceshine

License
-------

GNU General Public License v3.0
https://www.gnu.org/licenses/gpl-3.0.txt

28.08.2021
