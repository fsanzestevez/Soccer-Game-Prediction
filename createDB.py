# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 13:53:36 2020

@author: Fran
"""

import pandas as pd
import json
import os 


path_list = []
for (root, dirs, files) in os.walk('data'):
    for f in files:
        if f == 'season_stats.json':
            path = root + '/' + f
            path_list.append(path)

    
def loadData(path_list):
    # path = 'data/datafile/season17-18/season_stats.json'
    cols = ['Date', 'Team', 'Player_1', 'Player_2', 'Player_3', 'Player_4',
            'Player_5', 'Player_6', 'Player_7', 'Player_8', 'Player_9',
            'Player_10', 'Player_11']
    lineups = pd.DataFrame(columns=cols)
    for path in path_list:
        d = json.load(open(path, 'r', encoding='utf-8'))
        lineups = seasonLineups(lineups, d)
        print(lineups.shape)
    
    return lineups
    

def seasonLineups(df, d):
    lineups = df.copy()    
    for game_id in d.keys():
        game = d[game_id]
        for team_id in game.keys():
            team = game[team_id]
            date = team['team_details']['date']
            team_name = team['team_details']['team_name']
            players_lst = []
            plyr_sts = team['Player_stats']
            for player, stats in plyr_sts.items():
                place = int(stats['Match_stats']['formation_place'])
                if place:
                    players_lst.append(player)
            if len(players_lst) < 11:
                print(len(players_lst))
                input()
            lineups = lineups.append({'Date': date, 'Team': team_name,
                                      'Player_1': players_lst[0],
                                      'Player_2': players_lst[1],
                                      'Player_3': players_lst[2],
                                      'Player_4': players_lst[3],
                                      'Player_5': players_lst[4],
                                      'Player_6': players_lst[5],
                                      'Player_7': players_lst[6],
                                      'Player_8': players_lst[7],
                                      'Player_9': players_lst[8],
                                      'Player_10': players_lst[9],
                                      'Player_11': players_lst[10]},
                                     ignore_index=True)
    return lineups


lineupDB = loadData(path_list)





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    