
use speechanalyzer

create table users (
UserID int NOT NULL AUTO_INCREMENT,
FirstName varchar(20) NOT NULL,
LastName varchar(20) NOT NULL,
EmailAddress nvarchar(80) NOT NULL,
UserPassword varchar(250) NOT NULL,
CreateDate datetime NOT NULL,
PRIMARY KEY(UserID),
UNIQUE(EmailAddress)
);




create table speechfact (
SpeechID int NOT NULL auto_increment,
UserID int NOT NULL,
SpeechName varchar(50),
SpeechText text,
LoadDateTime datetime,
TargetAudience varchar(50),
LeanStrength varchar(50),
SylCt int,
WordCt int,
SentCt int,
EstSpeechTime decimal(5, 2),
GradeLevelComp float,
Relatability float,
PRIMARY KEY(SpeechID),
FOREIGN KEY (UserID) REFERENCES users(userID)
);

Create table userlog (
UserID int NOT NULL,
LoginDtTime datetime,
IPAddress varchar(150),
FOREIGN KEY (UserID) REFERENCES users(userID)
);


create table speechemotion (
SpeechID int NOT NULL,
Anger DECIMAL(6,3),
Sadness DECIMAL(6,3),
Joy DECIMAL(6,3),
Fear DECIMAL(6,3),
Suprise DECIMAL(6,3),
FOREIGN KEY (SpeechID) REFERENCES speechfact(SpeechID)
);


create table speechpersonality (
SpeechID int NOT NULL,
Openness decimal(6,3),
Extraversion decimal(6,3),
Agreeableness decimal(6,3),
Conscientiousness decimal(6,3),
FOREIGN KEY (SpeechID) REFERENCES speechfact(SpeechID)
);

create table speechparty (
SpeechID int NOT NULL,
Libertarian decimal(6,3),
Conservative decimal(6,3),
Liberal decimal(6,3),
Green decimal(6,3),
FOREIGN KEY (SpeechID) REFERENCES speechfact(SpeechID)
);


create table speechsentiment (
SpeechID int NOT NULL,
SpeechSentiment decimal(6,3),
Sentiment1 decimal(6,3),
Sentiment2 decimal(6,3),
Sentiment3 decimal(6,3),
Sentiment4 decimal(6,3),
Sentiment5 decimal(6,3),
SlightlyPos decimal(6,3),
VeryPos decimal(6,3),
ExtremelyPos decimal(6,3),
SlightlyNeg decimal(6,3),
VeryNeg decimal(6,3),
ExtremelyNeg decimal(6,3),
Neutral decimal(6,3),
FOREIGN KEY (SpeechID) REFERENCES speechfact(SpeechID)
);

Create Table speechpersona (
SpeechID int NOT NULL,
Campaigner decimal(6,3),
Debater decimal(6,3),
Protagonist decimal(6,3),
Commander decimal(6,3),
Mediator decimal(6,3),
Logician decimal(6,3),
Advocate decimal(6,3),
Architect decimal(6,3),
Entertainer decimal(6,3),
Entrepreneur decimal(6,3),
Consul decimal(6,3),
Executive decimal(6,3),
Adventurer decimal(6,3),
Virtuoso decimal(6,3),
Defender decimal(6,3),
Logistician decimal(6,3),
Foreign KEY (speechID) REFERENCES speechfact(SpeechID)
);


Create table TwitterSearchTerms (
UserID int not null,
Term varchar(50),
Positive decimal(6,3),
Negative decimal(6,3),
Neutral decimal(6,3),
SearchDateTime datetime,
FOREIGN KEY (UserID) REFERENCES users(UserID)
);


Create table demlocations (
UserID int not null,
CityState varchar(100),
SearchDateTime datetime,
foreign key (UserID) REFERENCES users(UserID)
);