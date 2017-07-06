# -*- coding: utf-8 -*-
import configFile
from flask import render_template, flash, redirect, request, Flask, Response, abort
from app import app
from flask_wtf import Form
from wtforms.fields import StringField
from wtforms.widgets import TextArea
from .forms import PostForm, citySelect, twitter, ContactForm, LoginForm, SignUp
from textstat.textstat import textstat
import numpy as np
from nltk import tokenize
from textblob import TextBlob
import tweepy
from tweepy import OAuthHandler
import yweather
import indicoio
import re
import json
import pandas as pd
import requests
from nocache import nocache
from flask_mail import Mail, Message
import pymysql
import datetime
from collections import Counter
import math
from passlib.hash import sha256_crypt
import nltk.data





hostname = configFile.hostname
username = configFile.username
password = configFile.password
database = configFile.database

conn = pymysql.connect( host=hostname, user=username, passwd=password, db=database )

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)




def fleschNorm (x):
    if x >= 90:
        return 5.0
    elif x >= 80:
        return 6.0
    elif x >= 70:
        return 7.0
    elif x >= 60:
        return 8.5
    elif x >= 50:
        return 11.0
    elif x >= 30:
        return 13.0
    else:
        return 16.0
		
pos = []
neg = []
neut = []
names = []		
met = ["Positive", "Negative", "Neutral"]		
userid = []


app.config.update(
	MAIL_SERVER =configFile.MAIL_SERVER,
	MAIL_PORT = configFile.MAIL_PORT,
	MAIL_USE_SSL = configFile.MAIL_USE_SSL,
	MAIL_USERNAME = configFile.MAIL_USERNAME,
	MAIL_PASSWORD = configFile.MAIL_PASSWORD
	)
	
mail = Mail(app)	

		
class TwitterClient(object):
	'''
	Generic Twitter Class for sentiment analysis.
	'''
	def __init__(self):
		'''
		Class constructor or initialization method.
		'''
		# keys and tokens from the Twitter Dev Console
		consumer_key = configFile.consumer_key
		consumer_secret = configFile.consumer_secret
		access_token = configFile.access_token
		access_token_secret = configFile.access_token_secret
		# attempt authentication
		try:
			# create OAuthHandler object
			self.auth = OAuthHandler(consumer_key, consumer_secret)
			# set access token and secret
			self.auth.set_access_token(access_token, access_token_secret)
			# create tweepy API object to fetch tweets
			self.api = tweepy.API(self.auth)
		except:
			print("Error: Authentication Failed")
	def clean_tweet(self, tweet):
		return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
	def get_tweet_sentiment(self, tweet):
		analysis = TextBlob(self.clean_tweet(tweet))
		if analysis.sentiment.polarity > 0:
			return 'positive'
		elif analysis.sentiment.polarity == 0:
			return 'neutral'
		else:
			return 'negative'
	def get_tweets(self, query, count = 10):
		# empty list to store parsed tweets
		tweets = []
		try:
			# call twitter api to fetch tweets
			fetched_tweets = self.api.search(q = query, count = count)
			# parsing tweets one by one
			for tweet in fetched_tweets:
				# empty dictionary to store required params of a tweet
				parsed_tweet = {}
				# saving text of tweet
				parsed_tweet['text'] = tweet.text
				# saving sentiment of tweet
				parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
				# appending parsed tweet to tweets list
				if tweet.retweet_count > 0:
					# if tweet has retweets, ensure that it is appended only once
					if parsed_tweet not in tweets:
						tweets.append(parsed_tweet)
				else:
					tweets.append(parsed_tweet)
			# return parsed tweets
			return tweets
		except tweepy.TweepError as e:
			# print error (if any)
			print("Error : " + str(e))
def main(x):
	# creating object of TwitterClient Class
	api = TwitterClient()
	# calling function to get tweets
	tweets = api.get_tweets(query = x, count = 200)
	# picking positive tweets from tweets
	ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
	# percentage of positive tweets
	positive = 100*((len(ptweets))/len(tweets))
	# picking negative tweets from tweets
	ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
	# percentage of negative tweets
	negative = 100*((len(ntweets))/len(tweets))
	# percentage of neutral tweets
	neutral = 100*((len(tweets) - len(ntweets) - len(ptweets))/len(tweets))
	positives = round(positive,2)
	negatives = round(negative,2)
	neutrals = round(neutral,2)
	pos.append(positives)
	neg.append(negatives)
	neut.append(neutrals)
	names.append(str(x))



##########
# Routes #
##########

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
	return(render_template("index.html"))


@app.route('/load', methods=['GET', 'POST'])
def load():
    form=PostForm()
    return render_template("load.html", form=form)
	
	
	
@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
	speech = re.sub(r'''\\['"]''',"",request.form['body'])
	poliLean = request.form['party']
	polStrength = request.form['strength']
	speechName = request.form['speechName']
	sylCount = round(textstat.syllable_count(speech), 0)
	lexCount = round(textstat.lexicon_count(speech, False), 0)
	sentCount = round(textstat.sentence_count(speech), 0)
	flesch = textstat.flesch_reading_ease(speech)
	fleschkin = textstat.flesch_kincaid_grade(speech)
	fog = textstat.gunning_fog(speech)
	smog = textstat.smog_index(speech)
	auto = textstat.automated_readability_index(speech)
	coleman = textstat.coleman_liau_index(speech)
	linsear = textstat.linsear_write_formula(speech)
	dale = textstat.dale_chall_readability_score(speech)
	fleschReal = fleschNorm(flesch)
	speechTimeNum = round(float(lexCount)/100, 2)
	speechTime = str(speechTimeNum)+' minutes'
	grades = [fleschReal, fleschkin, smog, auto, coleman, linsear, dale]
	meanGradeLevel = round(np.average(grades), 2)
	x = {'SylCount': sylCount, 'Word Count':lexCount, 'Sentence Count':sentCount, 'Est Speech Time':speechTime, 'Grade Level Comprehension':meanGradeLevel}#, 'Flesch':flesch, 'FleschKinkaid':fleschkin, 'FOG':round(fog,2), 'SMOG':smog, 'AUTO':auto, 'Coleman':coleman, 'Linsear':round(linsear,2), 'Dale':dale, 'FleschReal': fleschReal}
	allCombined = json.dumps(x, ensure_ascii=False)
	allCombined = "["+str(allCombined)+"]"
	metrics = ['Flesch', 'FleschKinkaid', 'FOG', 'SMOG', 'AUTO', 'Coleman', 'Linsear', 'Dale', 'FleschReal']
	values = [flesch, fleschkin, fog, smog, auto, coleman, linsear, dale, fleschReal]

	#########################
	# Sentiment by Sentence #
	#########################
	
	xaxis = []
	yaxis = []
	p = speech
	speechList = tokenize.sent_tokenize(p) # Tokenization of speech for list of sentences

	a = 0
	for i in speechList:
		blob = TextBlob(speechList[a])
		sent = blob.sentiment.polarity
		xaxis.append(a)
		yaxis.append(sent)
		a += 1


	bins = np.linspace(-1, 1, 10)
	digitized = np.digitize(yaxis, bins)
	raw_data = {'sentiment': yaxis}
	df = pd.DataFrame(raw_data, columns = ['sentiment'])
	bins = np.linspace(-1, 1, 8)
	group_names = ['Extremely Negative', 'Very Negative', 'Slightly Negative', 'Neutral', 'Slightly Positive', 'Very Positive', 'Extremely Positive']
	categories = pd.cut(df['sentiment'], bins, labels = group_names)
	df['categories'] = pd.cut(df['sentiment'], bins, labels=group_names)
	cts = pd.value_counts(df['categories'])
	output = pd.Series.to_dict(cts)
	binNames = []
	values = []
	for k,v in output.items():
		binNames.append(k)
		values.append(np.asscalar(v))
	histogramData = json.dumps([{'binName': country, 'Value': wins} for country, wins in zip(binNames, values)])

			
	######################
	# Sentiment By Fifth #
	######################
	
	x = round((float(1)/float(5)*len(yaxis)), ndigits=0)
	x = int(x)
	firstFifth = round(np.average(yaxis[0:(x+1)]),2)
	secondFifth = round(np.average(yaxis[(x+2):(x*2+1)]),2)
	thirdFifth = round(np.average(yaxis[(x*2+2):(x*3+1)]),2)
	fourthFifth = round(np.average(yaxis[(x*3+2):(x*4+1)]),2)
	lastFifth = round(np.average(yaxis[(x*4+2):]),2)
	compa = [1, 2, 3, 4, 5]
	compb = [firstFifth, secondFifth, thirdFifth, fourthFifth, lastFifth]

		
	c = compb
	d = compa
	cd = json.dumps([{'Sentiment': country, 'Group': wins} for country, wins in zip(c, d)])
	
	######################
	# Political Spectrum #
	######################
	
	#affil = "Conservative"
	#strength = "Strong"
	pol = indicoio.political(speech)
	parties = [str(x) for x in pol]
	values = pol.values()
	politicalDict = dict(zip(parties, values))

	politicalDict2 = json.dumps([{'PartyName': country, 'Value': wins} for country, wins in zip(parties, values)])
	
	
	
	######################
	# Relatability Index #
	######################
	
	relatability = 100
	nationalLevel = 7.5
	fullBlob = TextBlob(speech)
	fullSent = fullBlob.sentiment.polarity

	if meanGradeLevel == nationalLevel:
		c = 0
	elif (meanGradeLevel-nationalLevel) < 0:
		if (meanGradeLevel-nationalLevel) >-2:
			c = 10*(meanGradeLevel - nationalLevel)
		else:
			c = 15*(meanGradeLevel - nationalLevel+2)
	else:
		if (meanGradeLevel - nationalLevel) < 2:
			c = 10*(meanGradeLevel - nationalLevel)
		else:
			c = -15*(meanGradeLevel - nationalLevel-2)


	if (float(firstFifth) + float(lastFifth))/2 > (float(secondFifth) + float(thirdFifth) + float(fourthFifth))/3:
		d = 0
	else:
		d = -20

	if fullSent < .5:
		e = -10
	else:
		e = 10

	polParty = request.form['party']
	stren = request.form['strength']
	
	if pol[polParty] < 0.5:
		if pol[polParty] < 0.3:
			v = -40
		else:
			v = -20
	else:
		if stren == 'Moderate' and pol[polParty] < 0.75:
			v = 20
		elif stren == 'Strong' and pol[polParty] < 0.75:
			v = 10
		elif stren == 'Moderate' and pol[polParty] >= 0.75:
			v = 10
		elif stren == 'Strong' and pol[polParty] >= 0.75:
			v = 20
		else:
			v = 0

	relatability = relatability + c + d + e + v
	
	




	pers = indicoio.personality(speech)

	labels = [str(x) for x in pers]
	vals = list(pers.values())
	politicalDict3 = json.dumps([{'Trait': country, 'Strength': wins} for country, wins in zip(labels, vals)])
	

	pers2 = indicoio.personas(speech)

	xvals2 = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
	labels2 = [str(x) for x in pers2]
	vals2 = list(pers2.values())
	
	personaDict = json.dumps([{'Persona': country, 'Strength': wins} for country, wins in zip(labels2, vals2)])
	
	emotion = indicoio.emotion(speech)
	emot = [str(x) for x in emotion]
	emotvals = list(emotion.values())
	emotionDict = json.dumps([{'Emotion': country, 'Value': wins} for country, wins in zip(emot, emotvals)])
	
	
	try:
		with conn.cursor() as cursor:
		# Create a new record
			sql = "INSERT INTO `speechFact` (`UserID`, `SpeechName`, `SpeechText`, `LoadDateTime`, `TargetAudience`, `LeanStrength`, `SylCt`, `WordCt`, `SentCt`, `EstSpeechTime`, `GradeLevelComp`, `Relatability`, `Sentiment`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
			speechPartySQL = "Insert into `speechparty` (`SpeechID`, `Libertarian`, `Conservative`, `Liberal`, `Green`) VALUES (%s, %s, %s, %s, %s)"
			speechEmotionSQL = "Insert into `speechemotion` (`SpeechID`, `Anger`, `Sadness`, `Joy`, `Fear`, `Surprise`) VALUES (%s, %s, %s, %s, %s, %s)"
			speechPersonalitySQL = "Insert into `speechpersonality` (`SpeechID`, `Openness`, `Extraversion`, `Agreeableness`, `Conscientiousness`) VALUES (%s, %s, %s, %s, %s)"
			speechPersonaSQL = "Insert into `speechpersona` (`SpeechID`, `Campaigner`, `Debater`, `Protagonist`, `Commander`, `Mediator`, `Logician`, `Advocate`, `Architect`, `Entertainer`, `Entrepreneur`, `Consul`, `Executive`, `Adventurer`, `Virtuoso`, `Defender`, `Logistician`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
			speechidSQL = "Select LAST_INSERT_ID()"
			cursor.execute(sql, (userid[0], speechName, speech, datetime.datetime.now(), request.form['party'], request.form['strength'], sylCount, lexCount, sentCount, speechTimeNum, meanGradeLevel.item(), relatability.item(), fullSent))
			cursor.execute(speechidSQL)
			speechid = cursor.fetchall()
			cursor.execute(speechPartySQL, (speechid, politicalDict['Libertarian'], politicalDict['Conservative'], politicalDict['Liberal'], politicalDict['Green']))
			cursor.execute(speechEmotionSQL, (speechid, emotion['anger'], emotion['sadness'], emotion['joy'], emotion['fear'], emotion['surprise']))
			cursor.execute(speechPersonalitySQL, (speechid, pers['openness'], pers['extraversion'], pers['agreeableness'], pers['conscientiousness']))
			cursor.execute(speechPersonaSQL, (speechid, pers2['campaigner'], pers2['debater'], pers2['protagonist'], pers2['commander'], pers2['mediator'], pers2['logician'], pers2['advocate'], pers2['architect'], pers2['entertainer'], pers2['entrepreneur'], pers2['consul'], pers2['executive'], pers2['adventurer'], pers2['virtuoso'], pers2['defender'], pers2['logistician']))
		conn.commit()
	finally:
		trash = 1
	
	

	sensitivity = float(request.form['sensitivity'])
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	tokenized = tokenizer.tokenize(speech)
 
 
	html = ""
 
	output = indicoio.political(tokenized)
 
	counterSent = 0
	 
	while counterSent < len(output):
		if counterSent > 0 and counterSent%12 == 0:
			html+=str('<br><br>')
		if output[counterSent]['Conservative']>sensitivity:
			html+= str("<mark class = 'conservative'>"+str(tokenized[counterSent])+'</mark>')
		elif output[counterSent]['Liberal']>sensitivity:
			html+=str("<mark class = 'liberal'>"+str(tokenized[counterSent])+'</mark>')
		elif output[counterSent]['Green']>sensitivity:
			html+=str("<mark class = 'green'>"+str(tokenized[counterSent])+'</mark>')
		elif output[counterSent]['Libertarian']>sensitivity:
			html+=str("<mark class = 'libertarian'>"+str(tokenized[counterSent])+'</mark>')
		else:
			html+=str(str(tokenized[counterSent]))
		counterSent += 1
	
	return render_template("analyze.html", allCombined = allCombined, cd=cd, politicalDict2 = politicalDict2, histogramData=histogramData, politicalDict3=politicalDict3, personaDict = personaDict, relatability = relatability, emotionDict = emotionDict, html = html, sylCount=sylCount, lexCount=lexCount, sentCount=sentCount, speechTime=speechTime, meanGradeLevel=meanGradeLevel)
	
	
	
	

@app.route('/demographics', methods=['GET', 'POST'])
def demographics():
	location = request.form['cityState']
	string= location
	for ch in [',','"',"'"]:
		if ch in string:
			string=string.replace(ch, '')

	clean = string.replace(' ','+')
	address = clean
 
	address = address.replace(" ","+")
	address = address.replace(',',"")
	geo = requests.get('https://maps.googleapis.com/maps/api/geocode/json?address='+address+'&key='+configFile.apikey)
	geo = geo.json()	
	lat = geo['results'][0]['geometry']['location']['lat']
	long = geo['results'][0]['geometry']['location']['lng']
	
	revGeo = requests.get('https://maps.googleapis.com/maps/api/geocode/json?latlng='+str(lat)+','+str(long)+'&key='+configFile.apikey)
	 
	revGeo = revGeo.json()
	
	
	newAddressNum = revGeo['results'][0]['address_components'][0]['long_name']
	newAddressStreet = revGeo['results'][0]['address_components'][1]['long_name']
	newAddressCity = revGeo['results'][0]['address_components'][3]['long_name']
	if len(revGeo['results'][0]['address_components'][4]['short_name']) == 2:
		newAddressState = revGeo['results'][0]['address_components'][4]['short_name']
	else:
		newAddressState = revGeo['results'][0]['address_components'][5]['short_name']

	newAddress = str(newAddressNum) +' '+ str(newAddressStreet)
	 
	address = newAddress.replace(' ', '+')
	
	r = requests.get('https://geocoding.geo.census.gov/geocoder/geographies/address?street='+address+'&city='+newAddressCity+'&state='+newAddressState+'&benchmark=Public_AR_Census2010&vintage=Census2010_Census2010&layers=14&format=json')
	actual = r.json()
	
	try:
		county = actual['result']['addressMatches'][0]['geographies']['Census Blocks'][0]['COUNTY']
		state = actual['result']['addressMatches'][0]['geographies']['Census Blocks'][0]['STATE']
	except IndexError:
		r = requests.get('http://nominatim.openstreetmap.org/reverse?format=json&lat='+str(lat)+'&lon='+str(long))
		actual = r.json()
		try:
			newAddressNum = actual['address']['house_number']
		except KeyError:
			newAddressNum = '111'
		newAddressStreet = actual['address']['road']
		newAddressState = actual['address']['state']
		try:
			newAddressCity = actual['address']['town']
		except (KeyError, IndexError) as e:
			newAddressCity = actual['address']['city']

		newAddress = str(newAddressNum) +' '+ str(newAddressStreet)
	 
		address = newAddress.replace(' ', '+')
	
		r = requests.get('https://geocoding.geo.census.gov/geocoder/geographies/address?street='+address+'&city='+newAddressCity+'&state='+newAddressState+'&benchmark=Public_AR_Census2010&vintage=Census2010_Census2010&layers=14&format=json')
		actual = r.json()
		county = actual['result']['addressMatches'][0]['geographies']['Census Blocks'][0]['COUNTY']
		state = actual['result']['addressMatches'][0]['geographies']['Census Blocks'][0]['STATE']
		

	
	#############
	# Work Type #
	#############

	q = requests.get('http://api.census.gov/data/2015/acs5?get=NAME,'+'B24080_001E'+','+'B24080_002E'+','+'B24080_003E'+','+'B24080_004E'+','+'B24080_005E'+','+'B24080_006E'+','+'B24080_007E'+','+'B24080_008E'+','+'B24080_009E'+','+'B24080_010E'+','+'B24080_011E'+','+'B24080_012E'+','+'B24080_013E'+','+'B24080_014E'+','+'B24080_015E'+','+'B24080_016E'+','+'B24080_017E'+','+'B24080_018E'+','+'B24080_019E'+','+'B24080_020E'+','+'B24080_021E'+'&for=county:'+str(county)+'&in=state:'+str(state)+'&key='+configFile.censusAPI)
	 
	actualQ = q.json()
	 
	totalPop = int(actualQ[1][1])
	privateCo = round((int(actualQ[1][4]) + int(actualQ[1][14]))/totalPop,2)*100
	selfEmploy = round((int(actualQ[1][5]) + int(actualQ[1][15]))/totalPop,2)*100
	notForProfit = round((int(actualQ[1][6]) + int(actualQ[1][16]))/totalPop,2)*100
	localGov = round((int(actualQ[1][7]) + int(actualQ[1][17]))/totalPop,2)*100
	stateGov = round((int(actualQ[1][8]) + int(actualQ[1][18]))/totalPop,2)*100
	fedGov = round((int(actualQ[1][9]) + int(actualQ[1][19]))/totalPop,2)*100
	selfEmplNotInc = round((int(actualQ[1][10]) + int(actualQ[1][20]))/totalPop,2)*100
	unpaid = round((int(actualQ[1][11]) + int(actualQ[1][21]))/totalPop,2)*100
	
	typeNames = ['Private Company', 'Self Employ Corp', 'NonProfit', 'Local Gov', 'State Gov', 'Fed Gov', 'Self Employ Not Corp', 'Unpaid']
	typeVals = [privateCo, selfEmploy, notForProfit, localGov, stateGov, fedGov, selfEmplNotInc, unpaid]
	typeDict = json.dumps([{'Work Type': country, '% of Population': wins} for country, wins in zip(typeNames, typeVals)])

	########
	# SNAP #
	########


	q2 = requests.get('http://api.census.gov/data/2015/acs5?get=NAME,'+'B22003_001E'+','+'B22003_002E'+','+'B22003_003E'+','+'B22003_004E'+','+'B22003_005E'+','+'B22003_006E'+','+'B22003_007E'+'&for=county:'+str(county)+'&in=state:'+str(state)+'&key='+configFile.censusAPI)
	 
	actualQ2 = q2.json()
	 
	totalPop2 = int(actualQ2[1][1])
	receivedSNAP = round(int(actualQ2[1][2])/totalPop2,2)*100
	notRecSNAP = round(int(actualQ2[1][5])/totalPop2,2)*100
	
	snapNames = ['Received Food Stamps/SNAP', 'Did Not Receive Food Stamps/SNAP']
	snapVals = [receivedSNAP, notRecSNAP]
	snapDict = json.dumps([{'Food Stamps/SNAP': country, '% of Population': wins} for country, wins in zip(snapNames, snapVals)])

	##########
	# Income #
	##########

	q3 = requests.get('http://api.census.gov/data/2015/acs5?get=NAME,'+'B19001_001E'+','+'B19001_002E'+','+'B19001_003E'+','+'B19001_004E'+','+'B19001_005E'+','+'B19001_006E'+','+'B19001_007E'+','+'B19001_008E'+','+'B19001_009E'+','+'B19001_010E'+','+'B19001_011E'+','+'B19001_012E'+','+'B19001_013E'+','+'B19001_014E'+','+'B19001_015E'+','+'B19001_016E'+','+'B19001_017E'+'&for=county:'+str(county)+'&in=state:'+str(state)+'&key='+configFile.censusAPI)
	 
	actualQ3 = q3.json()
	 
	totalPop3 = int(actualQ3[1][1])
	lessThan10k = round(int(actualQ3[1][2])/totalPop3,2)*100
	between10and20 = round((int(actualQ3[1][3]) + int(actualQ3[1][4]))/totalPop3,2)*100
	between20and30 = round((int(actualQ3[1][5]) + int(actualQ3[1][6]))/totalPop3,2)*100
	between30and40 = round((int(actualQ3[1][7]) + int(actualQ3[1][8]))/totalPop3,2)*100
	between40and50 = round((int(actualQ3[1][9]) + int(actualQ3[1][10]))/totalPop3,2)*100
	between50and60 = round(int(actualQ3[1][11])/totalPop3,2)*100
	between60and75 = round(int(actualQ3[1][12])/totalPop3,2)*100
	between75and100 = round(int(actualQ3[1][13])/totalPop3,2)*100
	between100and125 = round(int(actualQ3[1][14])/totalPop3,2)*100
	between125and150 = round(int(actualQ3[1][15])/totalPop3,2)*100
	between150and200 = round(int(actualQ3[1][16])/totalPop3,2)*100
	moreThan200 = round(int(actualQ3[1][17])/totalPop3,2)*100

	incomeNames = ['<10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-75', '75-100', '100-125', '125-150', '150-200', '>200']
	incomeVals = [lessThan10k, between10and20, between20and30, between30and40, between40and50, between50and60, between60and75, between75and100, between100and125, between125and150, between150and200, moreThan200]
	incomeDict = json.dumps([{'Annual Salary $ (in thousands)': country, '% of Population': wins} for country, wins in zip(incomeNames, incomeVals)])

	#################
	# Ed over 25 yrs#
	#################


	q4 = requests.get('http://api.census.gov/data/2015/acs5?get=NAME,'+'B15003_001E'+','+'B15003_002E'+','+'B15003_003E'+','+'B15003_004E'+','+'B15003_005E'+','+'B15003_006E'+','+'B15003_007E'+','+'B15003_008E'+','+'B15003_009E'+','+'B15003_010E'+','+'B15003_011E'+','+'B15003_012E'+','+'B15003_013E'+','+'B15003_014E'+','+'B15003_015E'+','+'B15003_016E'+','+'B15003_017E'+','+'B15003_018E'+','+'B15003_019E'+','+'B15003_020E'+','+'B15003_021E'+','+'B15003_022E'+','+'B15003_023E'+','+'B15003_024E'+','+'B15003_025E'+'&for=county:'+str(county)+'&in=state:'+str(state)+'&key='+configFile.censusAPI)
	 
	actualQ4 = q4.json()

	totalpop4 = int(actualQ4[1][1])
	noSchool = round(int(actualQ4[1][2])/totalpop4,2)*100
	someSchoolNoHigh = round((int(actualQ4[1][3])+int(actualQ4[1][4])+int(actualQ4[1][5])+int(actualQ4[1][6])+int(actualQ4[1][7])+int(actualQ4[1][8])+int(actualQ4[1][9])+int(actualQ4[1][10])+int(actualQ4[1][11])+int(actualQ4[1][12]))/totalpop4,2)*100
	someHighSchoolNoGrad = round((int(actualQ4[1][13])+int(actualQ4[1][14])+int(actualQ4[1][15])+int(actualQ4[1][16]))/totalpop4,2)*100
	highSchoolGrad = round((int(actualQ4[1][17])+int(actualQ4[1][18]))/totalpop4,2)*100
	collegeNoDegree = round((int(actualQ4[1][19])+int(actualQ4[1][20]))/totalpop4,2)*100
	associates = round(int(actualQ4[1][21])/totalpop4,2)*100
	bachelors = round(int(actualQ4[1][22])/totalpop4,2)*100
	masters = round(int(actualQ4[1][23])/totalpop4,2)*100
	professional = round(int(actualQ4[1][24])/totalpop4,2)*100
	phd = round(int(actualQ4[1][25])/totalpop4,2)*100

	edNames = ['No School', 'No High School', 'Some High School No Grad', 'High School Grad', 'College No Degree', 'Associates', 'Bachelors', 'Masters', 'Professional Degree', 'Phd']
	edVals = [noSchool, someSchoolNoHigh, someHighSchoolNoGrad, highSchoolGrad, collegeNoDegree, associates, bachelors, masters, professional, phd]
	edDict = json.dumps([{'Highest Education Level': country, '% of Population': wins} for country, wins in zip(edNames, edVals)])

	###############################
	# Marital Status: 15 and over #
	###############################
	
	q5 = requests.get('http://api.census.gov/data/2015/acs5?get=NAME,'+'B12001_001E'+','+'B12001_002E'+','+'B12001_003E'+','+'B12001_004E'+','+'B12001_005E'+','+'B12001_006E'+','+'B12001_007E'+','+'B12001_008E'+','+'B12001_009E'+','+'B12001_010E'+','+'B12001_011E'+','+'B12001_012E'+','+'B12001_013E'+','+'B12001_014E'+','+'B12001_015E'+','+'B12001_016E'+','+'B12001_017E'+','+'B12001_018E'+','+'B12001_019E'+'&for=county:'+str(county)+'&in=state:'+str(state)+'&key='+configFile.censusAPI)
	 
	actualQ5 = q5.json()

	totalpop5 = int(actualQ5[1][1])
	neverMarried = round((int(actualQ5[1][3])+int(actualQ5[1][12]))/totalpop5,2)*100
	marriedSpousePresent = round((int(actualQ5[1][5])+int(actualQ5[1][14]))/totalpop5,2)*100
	marriedSpouseAbsent = round((int(actualQ5[1][6])+int(actualQ5[1][15]))/totalpop5,2)*100
	widowed = round((int(actualQ5[1][9])+int(actualQ5[1][18]))/totalpop5,2)*100
	divorced = round((int(actualQ5[1][10])+int(actualQ5[1][19]))/totalpop5,2)*100

	maritalNames = ['Never Married', 'Married: Spouse Present', 'Married: Spouse Absent', 'Widowed', 'Divorced']
	maritalVals = [neverMarried, marriedSpousePresent, marriedSpouseAbsent, widowed, divorced]
	maritalDict = json.dumps([{'Marital Status': country, '% of Population': wins} for country, wins in zip(maritalNames, maritalVals)])


	#################
	# Speak English #
	#################

	q6 = requests.get('http://api.census.gov/data/2015/acs5?get=NAME,'+'B06007_001E'+','+'B06007_002E'+','+'B06007_003E'+','+'B06007_004E'+','+'B06007_005E'+','+'B06007_006E'+','+'B06007_007E'+','+'B06007_008E'+'&for=county:'+str(county)+'&in=state:'+str(state)+'&key='+configFile.censusAPI)
	 
	actualQ6 = q6.json()
	totalpop6 = int(actualQ6[1][1])
	englishOnly = round(int(actualQ6[1][2])/totalpop6,2)*100
	englishVeryWellSpanish = round(int(actualQ6[1][3])/totalpop6,2)*100
	englishPoorSpanish = round(int(actualQ6[1][4])/totalpop6,2)*100
	englishVeryWellOther = round(int(actualQ6[1][7])/totalpop6,2)*100
	englishPoorOther = round(int(actualQ6[1][8])/totalpop6,2)*100

	englishNames = ['English Only', 'Spanish: Good English', 'Spanish: Poor English', 'Other: Good English', 'Other: Poor English']
	englishVals = [englishOnly, englishVeryWellSpanish, englishPoorSpanish, englishVeryWellOther, englishPoorOther]
	englishDict = json.dumps([{'English Fluency': country, '% of Population': wins} for country, wins in zip(englishNames, englishVals)])

	########
	# Race #
	########

	q7 = requests.get('http://api.census.gov/data/2015/acs5?get=NAME,'+'B02001_001E'+','+'B02001_002E'+','+'B02001_003E'+','+'B02001_004E'+','+'B02001_005E'+','+'B02001_006E'+','+'B02001_007E'+','+'B02001_008E'+','+'B02001_009E'+','+'B02001_010E'+'&for=county:'+str(county)+'&in=state:'+str(state)+'&key='+configFile.censusAPI)
	 
	actualQ7 = q7.json()

	totalpop7 = int(actualQ7[1][1])
	white = round(int(actualQ7[1][2])/totalpop7, 2)*100
	black = round(int(actualQ7[1][3])/totalpop7, 2)*100
	amerIndOrAlaska = round(int(actualQ7[1][4])/totalpop7, 2)*100
	asian = round(int(actualQ7[1][5])/totalpop7, 2)*100
	poly = round(int(actualQ7[1][6])/totalpop7, 2)*100
	otherSingle = round(int(actualQ7[1][7])/totalpop7, 2)*100
	twoOrMore = round(int(actualQ7[1][8])/totalpop7, 2)*100

	raceNames = ['White', 'Black', 'American Indian/Alaskan', 'Asian', 'Polynesian', 'Other (Single Race)', 'Other (Two or More)']
	raceVals = [white, black, amerIndOrAlaska, asian, poly, otherSingle, twoOrMore]
	raceDict = json.dumps([{'Race': country, '% of Population': wins} for country, wins in zip(raceNames, raceVals)])

	#######
	# Age #
	#######

	q8 = requests.get('http://api.census.gov/data/2015/acs5?get=NAME,'+'B01001_001E'+','+'B01001_002E'+','+'B01001_003E'+','+'B01001_004E'+','+'B01001_005E'+','+'B01001_006E'+','+'B01001_007E'+','+'B01001_008E'+','+'B01001_009E'+','+'B01001_010E'+','+'B01001_011E'+','+'B01001_012E'+','+'B01001_013E'+','+'B01001_014E'+','+'B01001_015E'+','+'B01001_016E'+','+'B01001_017E'+','+'B01001_018E'+','+'B01001_019E'+','+'B01001_020E'+','+'B01001_021E'+','+'B01001_022E'+','+'B01001_023E'+','+'B01001_024E'+','+'B01001_025E'+','+'B01001_026E'+','+'B01001_027E'+','+'B01001_028E'+','+'B01001_029E'+','+'B01001_030E'+','+'B01001_031E'+','+'B01001_032E'+','+'B01001_033E'+','+'B01001_034E'+','+'B01001_035E'+','+'B01001_036E'+','+'B01001_037E'+','+'B01001_038E'+','+'B01001_039E'+','+'B01001_040E'+','+'B01001_041E'+','+'B01001_042E'+','+'B01001_043E'+','+'B01001_044E'+','+'B01001_045E'+','+'B01001_046E'+','+'B01001_047E'+','+'B01001_048E'+','+'B01001_049E'+'&for=county:'+str(county)+'&in=state:'+str(state)+'&key='+configFile.censusAPI)
	 
	actualQ8 = q8.json()
		  
	totalpop8 = int(actualQ8[1][1])
	under10 = round((int(actualQ8[1][3])+int(actualQ8[1][4])+int(actualQ8[1][27])+int(actualQ8[1][28]))/totalpop8,2)*100
	between10and20 = round((int(actualQ8[1][5])+int(actualQ8[1][6])+int(actualQ8[1][7])+int(actualQ8[1][8])+int(actualQ8[1][29])+int(actualQ8[1][30])+int(actualQ8[1][31])+int(actualQ8[1][32]))/totalpop8,2)*100
	between21and30 = round((int(actualQ8[1][9])+int(actualQ8[1][10])+int(actualQ8[1][11])+int(actualQ8[1][33])+int(actualQ8[1][34])+int(actualQ8[1][35]))/totalpop8,2)*100
	between31and40 = round((int(actualQ8[1][12])+int(actualQ8[1][13])+int(actualQ8[1][36])+int(actualQ8[1][37]))/totalpop8,2)*100
	between41and50 = round((int(actualQ8[1][14])+int(actualQ8[1][15])+int(actualQ8[1][38])+int(actualQ8[1][39]))/totalpop8,2)*100
	between51and60 = round((int(actualQ8[1][16])+int(actualQ8[1][17])+int(actualQ8[1][40])+int(actualQ8[1][41]))/totalpop8,2)*100
	between61and70 = round((int(actualQ8[1][18])+int(actualQ8[1][19])+int(actualQ8[1][20])+int(actualQ8[1][21])+int(actualQ8[1][42])+int(actualQ8[1][43])+int(actualQ8[1][44])+int(actualQ8[1][45]))/totalpop8,2)*100
	between71and80 = round((int(actualQ8[1][22])+int(actualQ8[1][23])+int(actualQ8[1][46])+int(actualQ8[1][47]))/totalpop8,2)*100
	over80 = round((int(actualQ8[1][24])+int(actualQ8[1][25])+int(actualQ8[1][48])+int(actualQ8[1][49]))/totalpop8,2)*100

	ageNames = ['Under 10', '10-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', 'Over 80']
	ageVals = [under10, between10and20, between21and30, between31and40, between41and50, between51and60, between61and70, between71and80, over80]
	ageDict = json.dumps([{'Age in Years': country, '% of Population': wins} for country, wins in zip(ageNames, ageVals)])

	############
	# Ancestry #
	############

	q9 = requests.get('http://api.census.gov/data/2015/acs5?get=NAME,'+'B04006_001E'+','+'B04006_002E'+','+'B04006_003E'+','+'B04006_004E'+','+'B04006_005E'+','+'B04006_006E'+','+'B04006_007E'+','+'B04006_008E'+','+'B04006_009E'+','+'B04006_010E'+','+'B04006_011E'+','+'B04006_012E'+','+'B04006_013E'+','+'B04006_014E'+','+'B04006_015E'+','+'B04006_016E'+','+'B04006_017E'+','+'B04006_018E'+','+'B04006_019E'+','+'B04006_020E'+','+'B04006_021E'+','+'B04006_022E'+','+'B04006_023E'+','+'B04006_024E'+','+'B04006_025E'+','+'B04006_026E'+','+'B04006_027E'+','+'B04006_028E'+','+'B04006_029E'+','+'B04006_030E'+','+'B04006_031E'+','+'B04006_032E'+','+'B04006_033E'+','+'B04006_034E'+','+'B04006_035E'+','+'B04006_036E'+','+'B04006_037E'+','+'B04006_038E'+','+'B04006_039E'+','+'B04006_040E'+','+'B04006_041E'+','+'B04006_042E'+','+'B04006_043E'+','+'B04006_044E'+','+'B04006_045E'+','+'B04006_046E'+','+'B04006_047E'+','+'B04006_048E'+'&for=county:'+str(county)+'&in=state:'+str(state)+'&key='+configFile.censusAPI)
	actualQ9 = q9.json()
	q10 = requests.get('http://api.census.gov/data/2015/acs5?get=NAME,'+'B04006_049E'+','+'B04006_050E'+','+'B04006_051E'+','+'B04006_052E'+','+'B04006_053E'+','+'B04006_054E'+','+'B04006_055E'+','+'B04006_056E'+','+'B04006_057E'+','+'B04006_058E'+','+'B04006_059E'+','+'B04006_060E'+','+'B04006_061E'+','+'B04006_062E'+','+'B04006_063E'+','+'B04006_064E'+','+'B04006_065E'+','+'B04006_066E'+','+'B04006_067E'+','+'B04006_068E'+','+'B04006_069E'+','+'B04006_070E'+','+'B04006_071E'+','+'B04006_072E'+','+'B04006_073E'+','+'B04006_074E'+','+'B04006_075E'+','+'B04006_076E'+','+'B04006_077E'+','+'B04006_078E'+','+'B04006_079E'+','+'B04006_080E'+','+'B04006_081E'+','+'B04006_082E'+','+'B04006_083E'+','+'B04006_084E'+','+'B04006_085E'+','+'B04006_086E'+','+'B04006_087E'+','+'B04006_088E'+','+'B04006_089E'+','+'B04006_090E'+','+'B04006_091E'+','+'B04006_092E'+','+'B04006_093E'+','+'B04006_094E'+','+'B04006_095E'+'&for=county:'+str(county)+'&in=state:'+str(state)+'&key='+configFile.censusAPI)
	actualQ10 = q10.json()
	q11 = requests.get('http://api.census.gov/data/2015/acs5?get=NAME,'+'B04006_096E'+','+'B04006_097E'+','+'B04006_098E'+','+'B04006_099E'+','+'B04006_100E'+','+'B04006_101E'+','+'B04006_102E'+','+'B04006_103E'+','+'B04006_104E'+','+'B04006_105E'+','+'B04006_106E'+','+'B04006_107E'+','+'B04006_108E'+','+'B04006_109E'+'&for=county:'+str(county)+'&in=state:'+str(state)+'&key='+configFile.censusAPI)
	actualQ11 = q11.json()
	totalpop9 = int(actualQ9[1][1])

	afghan = round(int(actualQ9[1][2])/totalpop9,2)*100
	Albanian= round(int(actualQ9[1][3])/totalpop9,2)*100
	Alsatian= round(int(actualQ9[1][4])/totalpop9,2)*100
	American= round(int(actualQ9[1][5])/totalpop9,2)*100
	Egyptian= round(int(actualQ9[1][7])/totalpop9,2)*100
	Iraqi= round(int(actualQ9[1][8])/totalpop9,2)*100
	Jordanian= round(int(actualQ9[1][9])/totalpop9,2)*100
	Lebanese= round(int(actualQ9[1][10])/totalpop9,2)*100
	Moroccan= round(int(actualQ9[1][11])/totalpop9,2)*100
	Palestinian= round(int(actualQ9[1][12])/totalpop9,2)*100
	Syrian= round(int(actualQ9[1][13])/totalpop9,2)*100
	Arab= round(int(actualQ9[1][14])/totalpop9,2)*100
	OtherArab= round(int(actualQ9[1][15])/totalpop9,2)*100
	Armenian= round(int(actualQ9[1][16])/totalpop9,2)*100
	Assyrian= round(int(actualQ9[1][17])/totalpop9,2)*100
	Australian= round(int(actualQ9[1][18])/totalpop9,2)*100
	Austrian= round(int(actualQ9[1][19])/totalpop9,2)*100
	Basque= round(int(actualQ9[1][20])/totalpop9,2)*100
	Belgian= round(int(actualQ9[1][21])/totalpop9,2)*100
	Brazilian= round(int(actualQ9[1][22])/totalpop9,2)*100
	British= round(int(actualQ9[1][23])/totalpop9,2)*100
	Bulgarian= round(int(actualQ9[1][24])/totalpop9,2)*100
	Cajun= round(int(actualQ9[1][25])/totalpop9,2)*100
	Canadian= round(int(actualQ9[1][26])/totalpop9,2)*100
	CarpathoRusyn= round(int(actualQ9[1][27])/totalpop9,2)*100
	Celtic= round(int(actualQ9[1][28])/totalpop9,2)*100
	Croatian= round(int(actualQ9[1][29])/totalpop9,2)*100
	Cypriot= round(int(actualQ9[1][30])/totalpop9,2)*100
	Czech= round(int(actualQ9[1][31])/totalpop9,2)*100
	Czechoslovakian= round(int(actualQ9[1][32])/totalpop9,2)*100
	Danish= round(int(actualQ9[1][33])/totalpop9,2)*100
	Dutch= round(int(actualQ9[1][34])/totalpop9,2)*100
	EasternEuropean= round(int(actualQ9[1][35])/totalpop9,2)*100
	English= round(int(actualQ9[1][36])/totalpop9,2)*100
	Estonian= round(int(actualQ9[1][37])/totalpop9,2)*100
	European= round(int(actualQ9[1][38])/totalpop9,2)*100
	Finnish= round(int(actualQ9[1][39])/totalpop9,2)*100
	French = round(int(actualQ9[1][40])/totalpop9,2)*100
	FrenchCanadian= round(int(actualQ9[1][41])/totalpop9,2)*100
	German= round(int(actualQ9[1][42])/totalpop9,2)*100
	GermanRussian= round(int(actualQ9[1][43])/totalpop9,2)*100
	Greek= round(int(actualQ9[1][44])/totalpop9,2)*100
	Guyanese= round(int(actualQ9[1][45])/totalpop9,2)*100
	Hungarian= round(int(actualQ9[1][46])/totalpop9,2)*100
	Icelander= round(int(actualQ9[1][47])/totalpop9,2)*100
	Iranian= round(int(actualQ9[1][48])/totalpop9,2)*100
	Irish= round(int(actualQ10[1][1])/totalpop9,2)*100
	Israeli= round(int(actualQ10[1][2])/totalpop9,2)*100
	Italian= round(int(actualQ10[1][3])/totalpop9,2)*100
	Latvian= round(int(actualQ10[1][4])/totalpop9,2)*100
	Lithuanian= round(int(actualQ10[1][5])/totalpop9,2)*100
	Luxemburger= round(int(actualQ10[1][6])/totalpop9,2)*100
	Macedonian= round(int(actualQ10[1][7])/totalpop9,2)*100
	Maltese= round(int(actualQ10[1][8])/totalpop9,2)*100
	NewZealander= round(int(actualQ10[1][9])/totalpop9,2)*100
	NorthernEuropean= round(int(actualQ10[1][10])/totalpop9,2)*100
	Norwegian= round(int(actualQ10[1][11])/totalpop9,2)*100
	PennsylvaniaGerman= round(int(actualQ10[1][12])/totalpop9,2)*100
	Polish= round(int(actualQ10[1][13])/totalpop9,2)*100
	Portuguese= round(int(actualQ10[1][14])/totalpop9,2)*100
	Romanian= round(int(actualQ10[1][15])/totalpop9,2)*100
	Russian= round(int(actualQ10[1][16])/totalpop9,2)*100
	Scandinavian= round(int(actualQ10[1][17])/totalpop9,2)*100
	ScotchIrish= round(int(actualQ10[1][18])/totalpop9,2)*100
	Scottish= round(int(actualQ10[1][19])/totalpop9,2)*100
	Serbian= round(int(actualQ10[1][20])/totalpop9,2)*100
	Slavic= round(int(actualQ10[1][21])/totalpop9,2)*100
	Slovak= round(int(actualQ10[1][22])/totalpop9,2)*100
	Slovene= round(int(actualQ10[1][23])/totalpop9,2)*100
	SovietUnion= round(int(actualQ10[1][24])/totalpop9,2)*100
	CapeVerdean= round(int(actualQ10[1][26])/totalpop9,2)*100
	Ethiopian= round(int(actualQ10[1][27])/totalpop9,2)*100
	Ghanaian= round(int(actualQ10[1][28])/totalpop9,2)*100
	Kenyan= round(int(actualQ10[1][29])/totalpop9,2)*100
	Liberian= round(int(actualQ10[1][30])/totalpop9,2)*100
	Nigerian= round(int(actualQ10[1][31])/totalpop9,2)*100
	Senegalese= round(int(actualQ10[1][32])/totalpop9,2)*100
	SierraLeonean= round(int(actualQ10[1][33])/totalpop9,2)*100
	Somali= round(int(actualQ10[1][34])/totalpop9,2)*100
	SouthAfrican= round(int(actualQ10[1][35])/totalpop9,2)*100
	Sudanese= round(int(actualQ10[1][36])/totalpop9,2)*100
	Ugandan= round(int(actualQ10[1][37])/totalpop9,2)*100
	Zimbabwean= round(int(actualQ10[1][38])/totalpop9,2)*100
	African= round(int(actualQ10[1][39])/totalpop9,2)*100
	OtherSubsaharanAfrican= round(int(actualQ10[1][40])/totalpop9,2)*100
	Swedish= round(int(actualQ10[1][41])/totalpop9,2)*100
	Swiss= round(int(actualQ10[1][42])/totalpop9,2)*100
	Turkish= round(int(actualQ10[1][43])/totalpop9,2)*100
	Ukrainian= round(int(actualQ10[1][44])/totalpop9,2)*100
	Welsh= round(int(actualQ10[1][45])/totalpop9,2)*100
	Bahamian= round(int(actualQ10[1][47])/totalpop9,2)*100
	Barbadian= round(int(actualQ11[1][1])/totalpop9,2)*100
	Belizean= round(int(actualQ11[1][2])/totalpop9,2)*100
	Bermudan= round(int(actualQ11[1][3])/totalpop9,2)*100
	BritishWestIndian= round(int(actualQ11[1][4])/totalpop9,2)*100
	DutchWestIndian= round(int(actualQ11[1][5])/totalpop9,2)*100
	Haitian= round(int(actualQ11[1][6])/totalpop9,2)*100
	Jamaican= round(int(actualQ11[1][7])/totalpop9,2)*100
	TrinidadianTobagonian= round(int(actualQ11[1][8])/totalpop9,2)*100
	USVirginIslander= round(int(actualQ11[1][9])/totalpop9,2)*100
	WestIndian= round(int(actualQ11[1][10])/totalpop9,2)*100
	OtherWestIndian= round(int(actualQ11[1][11])/totalpop9,2)*100
	Yugoslavian= round(int(actualQ11[1][12])/totalpop9,2)*100
	Othergroups= round(int(actualQ11[1][13])/totalpop9,2)*100
	Unclassified= round(int(actualQ11[1][14])/totalpop9,2)*100

	ancestryNamesInter = ['Afghan', 'Albanian', 'Alsatian', 'American', 'Egyptian', 'Iraqi', 'Jordanian', 'Lebanese', 'Moroccan', 'Palestinian', 'Syrian', 'Arab', 'Other Arab', 'Armenian', 'Assyrian', 'Australian', 'Austrian', 'Basque', 'Belgian', 'Brazilian', 'British', 'Bulgarian', 'Cajun', 'Canadian', 'Carpatho Rusyn', 'Celtic', 'Croatian', 'Cypriot', 'Czech', 'Czechoslovakian', 'Danish', 'Dutch', 'Eastern European', 'English', 'Estonian', 'European', 'Finnish', 'French', 'French Canadian', 'German', 'German Russian', 'Greek', 'Guyanese', 'Hungarian', 'Icelander', 'Iranian', 'Irish', 'Israeli', 'Italian', 'Latvian', 'Lithuanian', 'Luxemburger', 'Macedonian', 'Maltese', 'New Zealander', 'Northern European', 'Norwegian', 'Pennsylvania German', 'Polish', 'Portuguese', 'Romanian', 'Russian', 'Scandinavian', 'Scotch-Irish', 'Scottish', 'Serbian', 'Slavic', 'Slovak', 'Slovene', 'Soviet Union', 'Cape Verdean', 'Ethiopian', 'Ghanaian', 'Kenyan', 'Liberian', 'Nigerian', 'Senegalese', 'Sierra Leonean', 'Somali', 'South African', 'Sudanese', 'Ugandan', 'Zimbabwean', 'African', 'Other Subsaharan African', 'Swedish', 'Swiss', 'Turkish', 'Ukrainian', 'Welsh', 'Bahamian', 'Barbadian', 'Belizean', 'British West Indian', 'Dutch West Indian', 'Haitian', 'Jamaican', 'Trinidadian and Tobagonian', 'U.S. Virgin Islander', 'West Indian', 'Other West Indian', 'Yugoslavian', 'Other Groups', 'Unclassified']
	ancestryValsInter = [afghan, Albanian, Alsatian, American, Egyptian, Iraqi, Jordanian, Lebanese, Moroccan, Palestinian, Syrian, Arab, OtherArab, Armenian, Assyrian, Australian, Austrian, Basque, Belgian, Brazilian, British, Bulgarian, Cajun, Canadian, CarpathoRusyn, Celtic, Croatian, Cypriot, Czech, Czechoslovakian, Danish, Dutch, EasternEuropean, English, Estonian, European, Finnish, French, FrenchCanadian, German, GermanRussian, Greek, Guyanese, Hungarian, Icelander, Iranian, Irish, Israeli, Italian, Latvian, Lithuanian, Luxemburger, Macedonian, Maltese, NewZealander, NorthernEuropean, Norwegian, PennsylvaniaGerman, Polish, Portuguese, Romanian, Russian, Scandinavian, ScotchIrish, Scottish, Serbian, Slavic, Slovak, Slovene, SovietUnion, CapeVerdean, Ethiopian, Ghanaian, Kenyan, Liberian, Nigerian, Senegalese, SierraLeonean, Somali, SouthAfrican, Sudanese, Ugandan, Zimbabwean, African, OtherSubsaharanAfrican, Swedish, Swiss, Turkish, Ukrainian, Welsh, Bahamian, Barbadian, Belizean, BritishWestIndian, DutchWestIndian, Haitian, Jamaican, TrinidadianTobagonian, USVirginIslander, WestIndian, OtherWestIndian, Yugoslavian, Othergroups, Unclassified]
	ancestryNames = []
	ancestryVals = []

	cc = 0
	while cc <len(ancestryValsInter):
		if ancestryValsInter[cc] > 0.005:
			ancestryNames.append(ancestryNamesInter[cc])
			ancestryVals.append(ancestryValsInter[cc])
		cc += 1

	ancestryDict = json.dumps([{'Ancestry': country, '% of Population': wins} for country, wins in zip(ancestryNames, ancestryVals)])
	ancestryDict = json.dumps([{'Ancestry': country, '% of Population': wins} for country, wins in zip(ancestryNames, ancestryVals)])
	
		
	#####################
	# Block Group Stats #
	#####################
	 
	#r3 = requests.get(('http://www.datasciencetoolkit.org/coordinates2statistics/'+str(lat)+'%2c'+str(long)))
	#r3 = r3.text
	#a3 = json.loads(r3)
	 
	#popDens = a3[0]['statistics']['population_density']['value']
	#households = a3[0]['statistics']['us_households']['value']
	#lingSep = a3[0]['statistics']['us_households_linguistically_isolated']['value']
	#singleMother = a3[0]['statistics']['us_households_single_mothers']['value']
	#housingUnits = a3[0]['statistics']['us_housing_units']['value']
	#housing1950To1969 = a3[0]['statistics']['us_housing_units_1950_to_1969']['value']
	#housing1970To1989 = a3[0]['statistics']['us_housing_units_1970_to_1989']['value']
	#housingAfter1990 = a3[0]['statistics']['us_housing_units_after_1990']['value']
	#housingBefore1950 = a3[0]['statistics']['us_housing_units_before_1950']['value']
	#noVehicle = a3[0]['statistics']['us_housing_units_no_vehicle']['value']
	#housingOccupied = a3[0]['statistics']['us_housing_units_occupied']['value']
	#housingOnePerson = a3[0]['statistics']['us_housing_units_one_person']['value']
	#housingOwnerOcc = a3[0]['statistics']['us_housing_units_owner_occupied']['value']
	#population = a3[0]['statistics']['us_population']['value']
	#asian = a3[0]['statistics']['us_population_asian']['value']
	#bachDegree = a3[0]['statistics']['us_population_bachelors_degree']['value']
	#black = a3[0]['statistics']['us_population_black_or_african_american']['value']
	#blackNoHispanic = a3[0]['statistics']['us_population_black_or_african_american_not_hispanic']['value']
	#age18To24 = a3[0]['statistics']['us_population_eighteen_to_twenty_four_years_old']['value']
	#age5To17 = a3[0]['statistics']['us_population_five_to_seventeen_years_old']['value']
	#foreignBorn = a3[0]['statistics']['us_population_foreign_born']['value']
	#hispanic = a3[0]['statistics']['us_population_hispanic_or_latino']['value']
	#lowIncome = a3[0]['statistics']['us_population_low_income']['value']
	#pacIsland = a3[0]['statistics']['us_population_native_hawaiian_and_other_pacific_islander']['value']
	#age1To4 = a3[0]['statistics']['us_population_one_to_four_years_olds']['value']
	#ageOver79 = a3[0]['statistics']['us_population_over_seventy_nine_years_old']['value']
	#poverty = a3[0]['statistics']['us_population_poverty']['value']
	#severePoverty = a3[0]['statistics']['us_population_severe_poverty']['value']
	#age65To79 = a3[0]['statistics']['us_population_sixty_five_to_seventy_nine_years_old']['value']
	#age25To64 = a3[0]['statistics']['us_population_twenty_five_to_sixty_four_years_old']['value']
	#ageUnder1 = a3[0]['statistics']['us_population_under_one_year_old']['value']
	#white = a3[0]['statistics']['us_population_white']['value']
	#whiteNoHisp = a3[0]['statistics']['us_population_white_not_hispanic']['value']
	
	#ageMetric = ['A: Age 1 To 4', 'B: Age 5 To 17', 'C: Age 18 To 24', 'D: Age 25 To 64', 'E: Age 65 To 79', 'F: Age Over 79']
	#ageVal = [round(age1To4*100,2), round(age5To17*100,2), round(age18To24*100,2), round(age25To64*100,2), round(age65To79*100,2), round(ageOver79*100,2)]
	
	#ageDict = json.dumps([{'Age': country, 'Value': wins} for country, wins in zip(ageMetric, ageVal)])
	
	#raceMetric = ['Asian', 'Black', 'Hispanic', 'Polynesian', 'White']
	#raceVal = [round(asian*100,2), round(blackNoHispanic*100,2), round(hispanic*100,2), round(pacIsland*100,2), round(whiteNoHisp*100,2)]
	#raceDict = json.dumps([{'Race': country, '% of Population': wins} for country, wins in zip(raceMetric, raceVal)])
	
	
	
	
	
	
	###########
	# Twitter #
	###########
	
	
	#client = yweather.Client()
	#woeid = client.fetch_woeid(string)

	# OAuth process, using the keys and tokens
	#auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	#auth.set_access_token(access_token, access_token_secret)
	#api = tweepy.API(auth)
	#trends1 = api.trends_place(woeid)
	#data = trends1[0]
	# grab the trends
	#trends = data['trends']
	# grab the name from each trend
	#names = [str(trend['name']) for trend in trends]
	
	#trends = []

	#for i in names:
	#	a = i.encode('utf-8')
	#	trends.append(a)

	#trends = trends[:10]
	
	#aa = [1, 2, 3, 4, 5,6,7,8,9,10]
	#bb = [str(trends[0]).replace("b'",""),str(trends[1]).replace("b'",""), str(trends[2]).replace("b'",""), str(trends[3]).replace("b'",""), str(trends[4]).replace("b'",""), str(trends[5]).replace("b'",""), str(trends[6]).replace("b'",""), str(trends[7]).replace("b'",""), str(trends[8]).replace("b'",""), str(trends[9]).replace("b'","")]
	#twitterDict = json.dumps([{'Metric': country, 'Value': wins} for country, wins in zip(aa, bb)])
	
	with conn.cursor() as cursor:
		# Create a new record
		demoLocationSQL = "Insert into `demlocations` (`UserID`, `CityState`, `SearchDateTime`) VALUES (%s, %s, %s)"
		cursor.execute(demoLocationSQL, (userid[0], location, datetime.datetime.now()))
	# connection is not autocommit by default. So you must commit to save
	# your changes.
	conn.commit()
	
	
	return render_template("demographics.html", typeDict=typeDict, snapDict = snapDict, incomeDict = incomeDict, edDict = edDict, maritalDict = maritalDict, englishDict = englishDict, raceDict=raceDict, ageDict = ageDict, ancestryDict=ancestryDict)#, twitterDict = twitterDict)	
	
	
@app.route('/twitterAnalyze', methods=['GET', 'POST'])
#@nocache
def twitterAnalyze():
	query1 = request.form['query1']
	query2 = request.form['query2']
	query3 = request.form['query3']
	main(query1)
	main(query2)
	main(query3)
	twitterDict2 = json.dumps([{'Name':names[0], "Measure": met[0], "Value":pos[0]},{'Name':names[1], "Measure": met[0], "Value":pos[1]},{'Name':names[2], "Measure": met[0], "Value":pos[2]},{'Name':names[0], "Measure": met[1], "Value":neg[0]},{'Name':names[1], "Measure": met[1], "Value":neg[1]},{'Name':names[2], "Measure": met[1], "Value":neg[2]},{'Name':names[0], "Measure": met[2], "Value":neut[0]},{'Name':names[1], "Measure": met[2], "Value":neut[1]},{'Name':names[2], "Measure": met[2], "Value":neut[2]}])
	return render_template("twitterAnalyze.html", twitterDict2 = twitterDict2)
	
@app.route('/about', methods=['GET', 'POST'])
def about():
	return render_template("about.html")
	
	
	
@app.route('/contact', methods=['GET', 'POST'])
def contact():
	cForm = ContactForm()
	if request.method == 'POST':
		if cForm.validate() == False:
			flash('All fields are required.')
			return render_template('contact.html', cForm = cForm)
		else:
			msg = Message(cForm.subject.data, sender = str(cForm['email']), recipients=['email@email.com'])
			msg.body = """
			From: %s;
			Subject: %s;
			Email Address: %s;
			Message: %s
			""" % (cForm.name.data, cForm.subject.data, cForm.email.data, cForm.message.data)
			msg.html = """
			<b>From:</b> %s;
			<br>
			<strong>Subject:</strong> %s;
			<br>
			<h4>Email Address:</h4> %s;
			<br>
			<h4>Message:</h4> %s
			""" % (cForm.name.data, cForm.subject.data, cForm.email.data, cForm.message.data)
			mail.send(msg)
			return render_template('contact.html', success=True)
			
	elif request.method == 'GET':
		return render_template('contact.html', cForm=cForm)
		


@app.route('/signup', methods = ['GET', 'POST'])
def signup():
	signupForm = SignUp()
	return render_template("signup.html", signupForm = signupForm)
	
	
	
@app.route('/signupCheck', methods=['GET', 'POST'])
def signupCheck():
	signupForm = SignUp()
	LogForm = LoginForm()
	first = request.form['first']
	last = request.form['last']
	email = request.form['email']
	password = sha256_crypt.hash(request.form['password'])
	insertUser = "INSERT INTO `users` (`FirstName`, `LastName`, `EmailAddress`, `userPassword`, `CreateDate`) VALUES (%s, %s, %s, %s, %s)"
	try:
		with conn.cursor() as cursor:
			check = "Select UserID from `users` where EmailAddress = %s"
			id = cursor.execute(check, email)
			if LogForm.validate() == False:
				flash('All fields are required.')
				return render_template('signup.html', signupForm = signupForm)
			else:
				if id == 0:
					cursor.execute(insertUser, (first, last, email, password, datetime.datetime.now()))
					conn.commit()
					return render_template('login.html', LogForm = LogForm)
				else:
					flash("Email address already used")
					return redirect("signup", code = 302)
	finally:
		trash = 1


@app.route('/login', methods = ['GET', 'POST'])
def login():
	LogForm = LoginForm()
	return render_template('login.html', LogForm = LogForm)
	

	
@app.route('/loginCheck', methods=['GET', 'POST'])
def loginCheck():
	email = request.form['email']
	password = request.form['password']
	IPAddress = request.environ['REMOTE_ADDR']
	userLogInsert = 'Insert into `userlog` (`UserID`, `LoginDtTime`, `IPAddress`) VALUES (%s, %s, %s)'
	try:
		with conn.cursor() as cursor:
			checklogin = "Select UserID from `users` where EmailAddress = %s"
			idInter = cursor.execute(checklogin, email)
			ident = cursor.fetchall()
			ident = ident[0][0]
			checkPass = "Select UserPassword from `users` where userid = %s"
			passInter = cursor.execute(checkPass, ident)
			passw = cursor.fetchall()
			passw = passw[0][0]
			passCheck = sha256_crypt.verify(password, passw)
			if (idInter != 0) and (passCheck == True):
				cursor.execute(userLogInsert, (ident, datetime.datetime.now(), IPAddress))
				conn.commit()
				userid.append(ident)
				return redirect('home')
			else:
				flash("Incorrect Login Credentials")
				return redirect("login", code = 302)
	except IndexError:
		flash("Incorrect Login Credentials")
		return redirect("login", code = 302)

	

@app.route('/home', methods=['GET', 'POST'])
def home():
	demoForm=citySelect()
	twitterForm=twitter()
	try:
		userid[0]
		with conn.cursor() as cursor:
			username = 'select FirstName from `users` where UserID = %s'
			cursor.execute(username, userid)
			firstname = cursor.fetchall()
			firstname = '\n'.join(''.join(elems) for elems in firstname)
		return render_template("home.html", demoForm=demoForm, twitterForm=twitterForm, firstname = firstname)
	except IndexError:
		return redirect('login')
		
			
@app.route('/demoload', methods=['GET', 'POST'])
def demoload():
	demoForm=citySelect()
	if demoForm.validate() == False:
		flash('You must enter a location')
		return render_template('demoload.html', demoForm=demoForm)
	else:
		try:
			userid[0]
			return render_template("demoload.html", demoForm=citySelect())
		except IndexError:
			return redirect('login')
		
		
@app.route('/twitterload', methods=['GET', 'POST'])
def twitterload():
	twitterForm=twitter()
	try:
		userid[0]
		return render_template("twitterload.html", twitterForm = twitterForm)
	except IndexError:
		return redirect('login')
		
		
@app.route('/history', methods=['GET', 'POST'])
def history():
	with conn.cursor() as cursor:
		try:
			sentimentPull = "Select SpeechName, Sentiment, date_format(LoadDateTime, '%%m/%%d/%%Y %%h:%%i') as LoadDate from `speechfact` where UserID = %s"
			sentiment = cursor.execute(sentimentPull, userid)
			sentiment = cursor.fetchall()
			headers = [ x[0] for x in sentiment ]
			sentimentVal = [ x[1] for x in sentiment ]
			date = [ x[2] for x in sentiment ]
			sentDict = json.dumps([{'SpeechName': country, 'Sentiment': wins, 'Date': apples} for country, wins, apples in zip(headers, sentimentVal, date)])
			relatePull = "Select SpeechName, Relatability, date_format(LoadDateTime, '%%m/%%d/%%Y %%h:%%i') as LoadDate from `speechfact` where UserID = %s"
			relate = cursor.execute(relatePull, userid)
			relate = cursor.fetchall()
			relateHeaders = [ x[0] for x in relate ]
			relateVal = [ x[1] for x in relate ]
			relateDate = [ x[2] for x in relate ]
			relateDict = json.dumps([{'SpeechName': country, 'Relatability': wins, 'Date': apples} for country, wins, apples in zip(relateHeaders, relateVal, relateDate)])
			partyPull = "Select avg(Libertarian) as AvgLibert, avg(Conservative) as AvgCons, avg(Liberal) as AvgLib, avg(Green) as AvgGreen from `speechparty` a inner join `speechfact` b on a.speechid = b.speechID where b.UserID = %s"
			parties = cursor.execute(partyPull, userid)
			parties = cursor.fetchall()
			partyHeaders = [ 'Libertarian', 'Conservative', 'Liberal', 'Green' ]
			partyVals = [ float(x) for x in parties[0] ]
			partyDict = json.dumps([{'Party': country, 'AvgValue': wins} for country, wins in zip(partyHeaders, partyVals)])
			personalityPull = "Select avg(Openness), avg(Extraversion), avg(Agreeableness), avg(Conscientiousness) from `speechpersonality` a inner join `speechfact` b on a.speechid = b.speechID where b.UserID = %s"
			personalities = cursor.execute(personalityPull, userid)
			personalities = cursor.fetchall()
			personalityHeaders = [ 'Openness', 'Extraversion', 'Agreeableness', 'Conscientiousness' ]
			personalityVals = [ float(x) for x in personalities[0] ]
			personalityDict = json.dumps([{'Trait': country, 'AvgValue': wins} for country, wins in zip(personalityHeaders, personalityVals)])
			personaPull = "Select avg(Campaigner), avg(Debater), avg(Protagonist), avg(Commander), avg(Mediator), avg(Logician), avg(Advocate), avg(Architect), avg(Entertainer), avg(Entrepreneur), avg(Consul), avg(Executive), avg(Adventurer), avg(Virtuoso), avg(Defender), avg(Logistician) from `speechpersona` a inner join `speechfact` b on a.speechid = b.speechID where b.UserID = %s"
			personas = cursor.execute(personaPull, userid)
			personas = cursor.fetchall()
			personasHeaders = [ 'Campaigner', 'Debater', 'Protagonist', 'Commander', 'Mediator', 'Logician', 'Advocate', 'Architect', 'Entertainer', 'Entrepreneur', 'Consul', 'Executive', 'Adventurer', 'Virtuoso', 'Defender', 'Logistician' ]
			personasVals = [ float(x) for x in personas[0] ]
			personasDict = json.dumps([{'Persona': country, 'AvgValue': wins} for country, wins in zip(personasHeaders, personasVals)])
			emotionPull = "Select avg(Anger), avg(Sadness), avg(Joy), avg(Fear), avg(Surprise) from `speechemotion` a inner join `speechfact` b on a.speechid = b.speechID where b.UserID = %s"
			emotions = cursor.execute(emotionPull, userid)
			emotions = cursor.fetchall()
			emotionsHeaders = [ 'Anger', 'Sadness', 'Joy', 'Fear', 'Surprise' ]
			emotionsVals = [ float(x) for x in emotions[0] ]
			emotionsDict = json.dumps([{'Emotion': country, 'AvgValue': wins} for country, wins in zip(emotionsHeaders, emotionsVals)])
			textList = []
			sql = "SELECT `speechName`, `speechText` FROM `speechFact` WHERE `userID`=%s order by loaddatetime"
			cursor.execute(sql, (userid))
			result = cursor.fetchall()
			df = pd.DataFrame(list(result))
			z = 0
			speechNames = []
			length = len(df)

			while z < length:
				speechNames.append(df[0][z])
				z += 1
			a = 0
			while a < length:
				textList.append(df[1][a])
				a += 1
			cosines = [1.00]
			
			counter = 0
			while counter < (len(textList)-1):
				vector1 = text_to_vector(textList[counter])
				vector2 = text_to_vector(textList[counter+1])
				Cosine = round(get_cosine(vector1, vector2), 2)
				cosines.append(Cosine)
				counter +=1
					
					
			cosineDict = json.dumps([{'SpeechName': country, 'Similarity': wins} for country, wins in zip(speechNames, cosines)])
		except TypeError:
			sentDict = 0
			relateDict = 0
			partyDict = 0
			cosineDict = 0
			personalityDict = 0
			personasDict = 0
			emotionsDict = 0
	try:
		userid[0]
		return render_template("history.html", sentDict = sentDict, relateDict = relateDict, partyDict = partyDict, cosineDict = cosineDict, personalityDict=personalityDict, personasDict = personasDict, emotionsDict = emotionsDict)
	except IndexError:
		return redirect('login')

		
@app.route('/logout', methods=['GET', 'POST'])
def logout():
	del userid[:]
	return render_template('logout.html')
		
