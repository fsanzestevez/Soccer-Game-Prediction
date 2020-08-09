# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 13:53:36 2020

@author: Fran
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import process
import json
import os 

import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 100  # Set Duration To 1000 ms == 1 second
debug = False

class CreateDB():
    
    def __init__(self, folder):
        self.folder = folder
        self.uniquePlayers = set()
        self.player2id = {'Chicharito': 178224,
                          'Nolito': 199561 }
        self.eplClubs = {'QPR': 'Queens Park Rangers'}
        self.notFound = set()
        self.fullDB = self.loadData()
        self.fullDB.to_pickle('data/full_DB.pkl')
        
    def getSeasonDB(self, lineups, odds, players_Fifa):
        clubs = list(lineups['Team'].unique()) \
                + list(odds['HomeTeam'].unique())
        clubs = set(clubs)
        # NaN != NaN, so we remove NaN quickly.
        clubs = {x for x in clubs if x==x} 
        for club in clubs:
            if club in self.eplClubs:
                newClub = self.eplClubs[club]

            else:
                newClub = process.extractOne(club,
                                             players_Fifa['club'].unique())[0]
                self.eplClubs[club] = newClub
            if club != newClub:
                lineups.loc[lineups['Team'] == club,
                            'Team'] = newClub
                odds.loc[odds['HomeTeam'] == club, 'HomeTeam'] = newClub
                odds.loc[odds['AwayTeam'] == club, 'AwayTeam'] = newClub
        odds['Date'] = pd.to_datetime(odds['Date'])
        lineups['Date'] = pd.to_datetime(lineups['Date'])
        
        seasonDB = odds.merge(lineups, how='left',
                              left_on=['Date', 'HomeTeam'],
                              right_on=['Date', 'Team'],
                              suffixes=[None,'_Home'])
        seasonDB = seasonDB.merge(lineups, how='left',
                                  left_on=['Date', 'AwayTeam'],
                                  right_on=['Date', 'Team'],
                                  suffixes=[None, '_Away'])
        return seasonDB
                  
    
    def findPlayer(self, player, club, players_Fifa):
        if player in self.player2id:
            if self.player2id[player] == -1:
                return np.nan


            playerFifa = players_Fifa[players_Fifa['sofifa_id'] == \
                                      self.player2id[player]]

            try:
                overall = int(playerFifa['overall'])
            except TypeError: # ID not found in fifa dataset
                overall = np.nan
            return overall
        else:

            if club in self.eplClubs:
                club = self.eplClubs[club]

            else:
                newClub = process.extractOne(club,
                                          players_Fifa['club'].unique())[0]
                self.eplClubs[club] = newClub
                club = newClub
            
            clubDF = players_Fifa[players_Fifa['club'] == club]
            name, ratio, _ = process.extractOne(player, clubDF['long_name'])
            playerFifa = clubDF[clubDF['long_name'] == name]

            if (len(playerFifa) != 1) or ratio < 75:
                playerFifa = players_Fifa[players_Fifa['long_name'] == player]
            
            if len(playerFifa) == 1:
                self.player2id[player] = int(playerFifa['sofifa_id'])

                return int(playerFifa['overall'])
            
            if len(playerFifa) != 1:
                best = process.extractOne(player,
                                               players_Fifa['long_name'])[0]
                playerFifa = players_Fifa[players_Fifa['long_name'] == best]
                
                if len(playerFifa) == 1:
                    self.player2id[player] = int(playerFifa['sofifa_id'])
                    return int(playerFifa['overall'])
                
                if len(playerFifa) > 1:
                    club = process.extractOne(club,
                                               players_Fifa['club'])[0]
                    playerFifa = playerFifa[playerFifa['club'] == club]
                    if len(playerFifa) == 1:
                        self.player2id[player] = int(playerFifa['sofifa_id'])
                        return int(playerFifa['overall'])
                    
                if len(playerFifa) != 1:
                    self.player2id[player] = -1
                    self.notFound.add((player, best, club))
                    return np.nan
            
        return np.nan
                
        
    def seasonLineups(self, df, d, players_Fifa):
        '''
        This method receives a JSON object (dict) and an empty base 
        DataFrame which it will be filling with the relevant data from the 
        JSON object
    
        Parameters
        ----------
        df : TYPE pandas.DataFrame
            DESCRIPTION. Empty base DF that will be filled up. It should have 
            the columns needed.
        d : TYPE dict
            DESCRIPTION. JSON object with the raw information about the lineups
            for every game in the season
        players : TYPE pandas.DataFrame
            DESCRIPTION. Fifa Ratings of the players of the specific season
    
        Returns
        -------
        lineups : TYPE pd.DataFrame
            DESCRIPTION. Filled up DataFrame with all the starting players of 
            each game.
    
        '''
        winsound.Beep(frequency, duration)
        lineups = df.copy()
        for game_id in d.keys():
            game = d[game_id]
            for team_id in game.keys():

                team = game[team_id]
                date = team['team_details']['date']
                team_name = team['team_details']['team_name']
                plyr_sts = team['Player_stats']
                game_dict = {'Date': date, 
                             'Team': team_name}
                i = 1
                for player, stats in plyr_sts.items():
                    place = int(stats['Match_stats']['formation_place'])
                    if place:
                        game_dict['Player_'+str(i)] = player
                        overall = self.findPlayer(player, team_name,
                                                  players_Fifa)
                        game_dict['P_'+str(i)+'_Overall'] = overall
  
                        self.uniquePlayers.add((player, team_name))
                        i += 1
                lineups = lineups.append(game_dict, ignore_index=True)

        return lineups
    
    def loadData(self):
        '''
        
    
        Returns
        -------
        lineups : TYPE
            DESCRIPTION.
    
        '''
        # cols = ['Date', 'Team']
        # for i in range(1, 12):
        #     cols.append('Player_'+str(i))
        #     cols.append('P_'+str(i)+'_Overall')
        # lineups = pd.DataFrame(columns=cols)
        lineups = pd.DataFrame()
        seasonDB = pd.DataFrame()
        fullDB = pd.DataFrame()
        if debug:
            d = json.load(open('data/season14-15/season_stats.json', 'r',
                               encoding='utf-8'))
            players = pd.read_csv('data/season14-15/players_15.csv')
            odds = pd.read_csv('data/season14-15/EPL_14-15.csv')
            lineups = self.seasonLineups(lineups, d, players)
            seasonDB = self.getSeasonDB(lineups, odds,
                                                    players)
                        
            fullDB = fullDB.append(seasonDB)
        else:
            for (root, dirs, files) in os.walk(self.folder):
                for f in files:
                    if f == 'season_stats.json':
                        path = root + '/' + f
                        d = json.load(open(path, 'r', encoding='utf-8'))

                        players = pd.read_csv(root+'/players_'+root[-2:]+'.csv')
                        odds = pd.read_csv(root+'/EPL_'+root[-5:]+'.csv')
                        lineups = self.seasonLineups(lineups, d, players)

                        seasonDB = self.getSeasonDB(lineups, odds,
                                                    players)
                        
                        fullDB = fullDB.append(seasonDB)
        fullDB.reset_index(inplace=True)    
        return fullDB
               

                
                
                
                
                
                
    