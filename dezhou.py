from pypokerengine.utils.card_utils import gen_cards,estimate_hole_card_win_rate
#花色：C梅花  D方块 H红桃 S黑桃
#1-K对应 A 2 3 4 5 6 7 8 9 T J Q K
NB_SIMULATION = 1000#模拟次数
nb_player = 3#在场玩家数
community_card = ['D2','D7','CA']#场上公牌
hole_card = ['SA','DA']#手牌
win_rate = estimate_hole_card_win_rate(
	nb_simulation = NB_SIMULATION, 
	nb_player = nb_player, 
	hole_card = gen_cards(hole_card), 
	community_card = gen_cards(community_card))
print(win_rate)
