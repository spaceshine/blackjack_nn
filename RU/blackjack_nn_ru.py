# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam

"""
# Подготовка данных

Вспомогательные функции
"""

def get_cards():
  """
    Определяет значения карт дилера и игрока
  """
  dlr_opn_crd = np.random.randint(2, 11)
  dlr_crds = np.random.randint(2, 11) + dlr_opn_crd

  return np.random.randint(4, 21), \
         dlr_crds, \
         dlr_opn_crd

def get_data_from_cards(cards):
  """
    Выносит из значений карт нужную для обучения информацию
  """
  player_cards, dealer_cards, _ = cards

  player_cards += np.random.randint(2, 11)  # просчет след карты игрока
  # именно по ней выносится оценка: какое действие надо совершить

  while dealer_cards < 17:  # просчет до финальной карты дилера
    dealer_cards += np.random.randint(2, 11)
  
  if dealer_cards > 21:  # случай перебора дилера: брать было не обязательно
    return cards[0], cards[1], cards[2], -0.5, player_cards, dealer_cards
  
  if player_cards > 21:  # случай перебора игрока: не надо было брать
    return cards[0], cards[1], cards[2], -1, player_cards, dealer_cards
  
  elif player_cards >= dealer_cards:  # случай выигрыша: брать обязательно
    return cards[0], cards[1], cards[2], 1, player_cards, dealer_cards
  
  else:  # случай неперебора: брать желательно
    return cards[0], cards[1], cards[2], 0.9, player_cards, dealer_cards

"""Генерируем данные для обучения нейросети"""

data_size = 30000 # @param размер обуч.выборки

data_raw = np.array([get_data_from_cards(get_cards()) for _ in range(data_size)])

print(data_raw[:10])  # формирование сырой выборки данных

X = data_raw[:, (0, 2)]  # входные данные: сумма карт игрока, откр.карта дилера
Y = data_raw[:, 3]  # выходные значения: решение, принимаемое нс

print(X[:10], Y[:10], sep='\n\n')

"""
# Создание модели нс
"""

model = Sequential([
          InputLayer((2,)),             # входной слой: [сумма карт игрока, откр.карта дилера]
          Dense(32, activation='relu'), # средний слой: 32 нейрона - активация relu
          Dense(1, activation='tanh')   # выходн. слой: от -1 до 1 - решение нс (-1: остановиться, +1: брать)
])

model.compile(
    optimizer=Adam(0.001),
    loss='mean_squared_error',
    metrics=['accuracy']
)

model.summary()

"""Обучение модели нс"""

model.fit(X, Y, batch_size=32, epochs=25, verbose=0)

"""
# Используем нейросеть

Простой пример
"""

print(model.predict([[18, 11]]))  # сумма карт игрока: 18, откр.карта дилера: 11(туз)

"""Вспомогательные функции"""

def boolprint(bll, msg):  # печатает если условие True
  if bll:
    print(msg)

def play_blackjack(nn_model, bet=0, balance=1, show=True):
  """
    Нейросеть играет 
    1 партию в блекджек с автоматизированым дилером
  """
  player, dealer, dealer_open = get_cards()  # раздача карт

  boolprint(bll=show, msg=f'Игрок: {player}, Откр.дилера: {dealer_open}') 

  if player == 21:       # ситуация Блекджека
    if dealer_open < 10: 
      isBlackJack = 1
    else:
      isBlackJack = 0
  else:
    isBlackJack = -1

  # пока нс берет карты, их значения прибавляются к сумме
  while nn_model.predict([[player, dealer_open]]) >= 0 and player < 21:
      player += np.random.randint(2, 11)
      boolprint(bll=show, msg=f'Игрок: {player}, Откр.дилера: {dealer_open}')

  boolprint(bll=show, msg='----')  # дилер начинает брать карты

  boolprint(bll=show, msg=f'Дилер: {dealer}')

  # не останавливается пока не наберет 17 - правила
  while dealer < 17:
      dealer += np.random.randint(2, 11)
      boolprint(bll=show, msg=f'Дилер: {dealer}')

  boolprint(bll=show, msg='----')  # итоги

  if isBlackJack == 1:
    boolprint(bll=show, msg=f'Победа игрока - Блекджек 3:2\nКарты: {player, dealer, dealer_open}')
    res = (1, player, dealer, dealer_open, balance + 1.5*bet, bet)
  
  elif isBlackJack == 0:
    boolprint(bll=show, msg=f'Победа игрока - Блекджек 1:1\nКарты: {player, dealer, dealer_open}')
    res = (1, player, dealer, dealer_open, balance + 1.0*bet, bet)
  
  elif dealer > 21:       # перебор у дилера
    boolprint(bll=show, msg=f'Победа игрока\nКарты: {player, dealer, dealer_open}')
    res = (1, player, dealer, dealer_open, balance + 1.0*bet, bet)
  
  elif player > 21:     # перебор игрока
    boolprint(bll=show, msg=f'Победа дилера(казино)\nКарты: {player, dealer, dealer_open}')
    res = (-1, player, dealer, dealer_open, balance - 1.0*bet, bet)
  
  elif dealer > player: # дилер > игрок
    boolprint(bll=show, msg=f'Победа дилера(казино)\nКарты: {player, dealer, dealer_open}')
    res = (-1, player, dealer, dealer_open, balance - 1.0*bet, bet)
  
  elif dealer == player:# дилер = игрок
    boolprint(bll=show, msg=f'Ничья\nКарты: {player, dealer, dealer_open}')
    res = (0, player, dealer, dealer_open, balance, bet)
  
  elif player > dealer: # дилер < игрок
    boolprint(bll=show, msg=f'Победа игрока\nКарты: {player, dealer, dealer_open}')
    res = (1, player, dealer, dealer_open, balance + 1.0*bet, bet)
  
  boolprint(bll=show, msg='\n-*---*-\n')  # окончание партии

  return res

"""Нс играет в блекджек"""

play_blackjack(model, 20, 100, show=True)

games =   500# @param кол-во тестовых игр

bal = 10000  # @param изначальный баланс

bet = 200    # @param ставка

his = []  # история результатов игр

for i in range(games):  # играет games игр 
  game = play_blackjack(model, bet, bal, show=False)  # играет 1 игру

  his.append(game) # добавляем рез-ты
  bal = game[-2]

his = np.array(his, dtype=int)
print(his[:5])

"""Статистика"""

print('Конечный  баланс:', bal)

print()

print('Процент поражений:', round(np.sum(his[:, 0] == -1) / games * 100, 1), '%')
print('Процент побед:    ', round(np.sum(his[:, 0] == 1) / games * 100, 1), '%')
print('Процент ничей:    ', round(np.sum(his[:, 0] == 0) / games * 100, 1), '%')

plt.plot(his[:, -2])  # график баланса
