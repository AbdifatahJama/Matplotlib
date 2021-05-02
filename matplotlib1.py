import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import datetime,date
from matplotlib import dates as mpl_dates

# # print(math.pi)
# # a = np.array([1,2,3,4,5,6,7])
# # b = a*2

# # # a.reshape([4,1])
# # # print(a.sum(axis = 0))

# # # plt.plot(a, b,'y')
# # # plt.xlabel('A')
# # # plt.ylabel('B')
# # # plt.title('A against B')
# # # plt.show()


# a = np.arange(0,2*math.pi,0.5*math.pi) # uses arange array in numpy to set range of angles on x axis
# b = np.sin(a) # sin of each angle

# plt.plot(a,b,'r')
# plt.xlabel('X:Angle in radians')
# plt.ylabel('Y')
# plt.title('Sin(x)')
# plt.show()

# a = np.arange(0,2*math.pi/2,math.pi/2)
# b = np.cos(a)

# plt.plot(a,b)
# plt.title('Cos wave')
# plt.xlabel('x-Radians')
# plt.ylabel('y')
# plt.show()

# # Qaudratic plot

# x = np.linspace(-10000,10000,1000000)
# y = x**2

# plt.plot(x,y)
# plt.title('Qaudratic Graph')
# plt.show()

# # Cubic graph
# x = np.linspace(-10000,10000,1000000)
# y = x**3
# plt.plot(x,y)
# plt.title('Cubic Layout')
# plt.show()
# x= np.linspace(0,5,2)
# print(x)


# x = np.linspace(-10,10,1)
# x = np.arange()
# y = 1/x**2
# plt.plot(x,y)
# plt.show()


# Simple plot

# x = np.arange(1,20)
# y = x**2
# plt.plot(x,y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('x y plot')
# plt.show()

# plt.subplot(1,2,1)
# plt.plot(x,y)
# plt.show()

# plt.subplot(1,2,2)
# plt.plot(x,y,'r')
# plt.show()

# # Oil Price data analysis

gas = pd.read_csv('gas_prices.csv.txt',delimiter = ',')
# print(gas)

# plt.plot(gas.loc[0:,'Year'],gas.loc[0:,'UK'], label = 'UK') # One way to parse in x,y data using pandas loc function using textual names for the specific column
# plt.plot(gas.Year,gas.USA,label = 'United States') # Or we can use the . method of the column or [''] for columns with more than one word ie: South Korea
# plt.plot(gas.Year,gas['South Korea'],label = 'South Korea')
# plt.title('Gas Prices in USD')
# plt.xlabel('Year')
# plt.ylabel('Gas price in USD')
# plt.legend()
# # plt.figure(figsize=(10,10)) # Default 10 width 8 height # Can also change dpi of image 
# plt.xticks(gas.Year[::3]) # starts at beggining and end at end and goes in steps of three
# plt.show()

# Currenly one of the main problems with the graph is that it shows half years ie:'2007.5'
# So we can use pandas list slicing to fix this to get full years using the xticks function to specify our ticks a bit more rather than the defualt ticks


## Need to add data point(Dot-Dash styles to emphasise data points)


# Plotting all of data in one go 

for country in gas:
  if country != 'Year':
    plt.plot(gas.Year,gas[country],label = country)
plt.xticks(gas.Year[::3].tolist()+[2011]) # Adjusts x axis ticks, we can also expand the x axis using the two list function 
plt.title('Price per gallon oil in USD',fontdict = {'fontweight':'bold','fontsize':26}) # Very important text modification using fontdict
plt.xlabel('Year')
plt.ylabel('US Dollars')
plt.legend()
plt.savefig('gas_prices.png',facecolor = 'b') # changed png background to blue, default is white
plt.show()

## Remember: Looping through dataset returns all the columns. In this case we used a simple for loop and removed the year column using an if statement condition

## Changing fontsize and font features in general using fontdict dictonary in text areas such as the x and y titles or the main title. 

## Saving graph image as a png file


# Fifa Data

fifa = pd.read_csv('fifa_data.csv.txt')
fifa.sort_values('Age',ascending = True)
# print(fifa.loc[0:,'Overall'].sort_values(ascending = True))
print(fifa)

# Fifa histogram - Looking at the frequencies of a specific overall
bin = [10,20,30,40,50,60,70,80,90,100]
plt.hist(fifa.Overall,bins = bin[3:],color = 'g') # adjusted bins so it starts at 40 as there no player below a 40 Overall
plt.xlabel('Player Overall',fontdict = {'fontweight':'bold'})
plt.ylabel('Player Frequencies',fontdict = {'fontweight':'bold'})
plt.title('Fifa Player Overalls',fontdict = {'fontsize':22})
plt.show()

# We can also adjust the bin of the histogram which is where the values of the axis are all an equal width apart

# Histogram with arsenal players
arsenal = fifa.loc[fifa['Club'] == 'Arsenal']
bin = [0,10,20,30,40,50,60,70,80,90,100]
plt.hist(arsenal.Overall,bins = bin[5:])
plt.xticks(bin[5:])
plt.show()

# Pie Chart - Right and Left foot
# This will put the proportion of left and right footed players in a pie chart

left = fifa.loc[fifa['Preferred Foot'] == 'Left'].count(axis = 0)[0]
print(left)

right = fifa.loc[fifa['Preferred Foot'] == 'Right'].count(axis = 0)[0]
print(right)

label = ['Left Footed','Right Footed'] #Labels anti clockwise
plt.pie([left,right],labels=label,startangle=180,explode = [0.1,0],shadow = True,colors = ['red','blue'],autopct ='%.2f%%') # first argument is the wedge size which calculates the percentage of each left and right foot, explode argument disconnects the first section by 0.1 away from the center
plt.show()

# Weights of Fifa player as a pie chart
# In order to work with the weight we would need to remove the lbs(pounds string) string 

weights = fifa.loc[0:,'Weight']
# print(weight[0][0:3])
list = []
for weight in weights: # Had to process the weight column to remove the 'lbs' string then put weight integers into a list, then convert the list into a pandas dataframe consisting of a single column called weight with 18k rows
  try:
    x = weight[0:3]
    x = int(x)
    list.append(x)
  
  except:
    print('Exception')
    break
  
print(list)

weight_df = pd.DataFrame({'Weight':list}) # List of weights conveted into panda dataframe with column weight and mutiple rows
print(weight_df)

# Calculation of number people in each weight category
light = weight_df.loc[weight_df['Weight']<125].count()[0]
light_medium = weight_df.loc[(weight_df['Weight']>=125) & (weight_df['Weight']<150)].count()[0]
medium = weight_df.loc[(weight_df['Weight']>=150) & (weight_df['Weight']<170)].count()[0]
medium_heavy = weight_df.loc[(weight_df['Weight']>=170) & (weight_df['Weight']<200)].count()[0]
heavy = weight_df.loc[weight_df['Weight']>200].count()[0]
print(heavy)


# # Quick historgram of weight

# plt.hist(weight_df['Weight'])
# plt.show()

# Pie chart of weight distrubution in weight

plt.pie([light,light_medium,medium,medium_heavy,heavy],labels=['Light','Light Medium','Medium','Medium Heavy','Heavy'],startangle=90,shadow = True,autopct = '%.2f%%',pctdistance=0.8,explode = [0.2,0.2,0.2,0.2,0.2])
plt.style.use('ggplot') # changes color scheme of the chart from default to 'ggplot' in this case, can be changed to different style schemes using documentation 
plt.savefig('Fifa_Weight.png')
plt.legend()
plt.title('Weight distrubution of players in fifa 18',fontdict = {'fontweight':'bold','fontsize':18})
plt.show()

# Box and whisker chart between two different teams - (Real madrid and Juventus)

madrid = fifa.loc[fifa['Club'] == 'Real Madrid' ]
madrid = madrid['Overall'].reset_index().drop(columns = ['index'])
print(madrid)

juve = fifa.loc[fifa['Club'] == 'Juventus' ]
juve = juve['Overall'].reset_index().drop(columns = ['index'])
print(juve)

norwich = fifa.loc[fifa.Club == 'Arsenal'].Overall # Alternative way to get overall column 
print(norwich)
plt.boxplot([madrid.Overall, juve.Overall,norwich],vert = False,labels = ['Real Madrid','Juventus','Norwich'])
plt.title('Players Overall box plot')
plt.xlabel('Overalls')
plt.ylabel('Team name')
plt.figure(figsize=(12,8)) # Reshaping size of image as it looks too compact at the moment 
plt.show()

# We can further change the styling of the box and whisper graphs

# Programming age and compensation graphs

prg = pd.read_csv('programmers.csv.txt',delimiter = ',')
print(prg)

# Age - All Devs Graph

plt.plot(prg.Age,prg.All_Devs,label = 'All Developers')
plt.title('Median compensation of all Python and Javascript Developers')
plt.xlabel('Age')
plt.ylabel('Compensation($USD)')
plt.style.use('seaborn')
plt.legend()
plt.show()

#Age - Python Graph

plt.plot(prg.Age,prg.Python,label = 'Python Developer')
plt.xlabel('Age')
plt.ylabel('Compensation($USD)')
plt.style.use('seaborn')
plt.legend()
plt.show()

# Age - Javascript Graph

plt.plot(prg.Age,prg['JavaScript'],label = 'JavaScript Developer')
plt.xlabel('Age')
plt.ylabel('Compensation($USD)')
plt.style.use('seaborn')
plt.legend()
plt.show()

# Age and all three plots on same graph

for dev in prg:
  if dev == 'Age':
    pass
  else:
    plt.plot(prg.Age,prg[dev],label = dev)


plt.xlabel('Age')
plt.ylabel('Compensation($USD)')
plt.title('All Three Compensation graphs')
plt.style.use('seaborn')
plt.legend()
plt.show()

# Box plot comparing Python and Javascript Devs

plt.boxplot([prg.Python,prg['JavaScript']],labels = ['Python','JavaScript'])
plt.title('Python and JavaScipt Compensation')
plt.xlabel('Language')
plt.ylabel('Median Compensation($USD)')
plt.style.context('dark_background')
plt.show()

# SubPlots
# Subplots allows secondary plots to be made of a data set. 
# Allows the data set graph illustrations to be more readable and clear


fig, (ax1,ax2) = plt.subplots(2,1,sharex = True) # One figure with 2 rows and 1 axis so 2 graphs vertically below each other
ax1.plot(prg.Age,prg.Python,label = 'Python',color = 'green')
ax2.plot(prg.Age,prg.JavaScript,label = 'JavaScript',color = 'red')
ax1.set_title('Python and JavaScript Compensation against Age') # Subplot have a slightly different syntax to plot it used set_title rather .title
# ax1.set_xlabel('Age') # Remove top column x axis so we only refer to axis on bottom graph
ax1.set_ylabel('Compensation($USD)') # Subplots uses set_ylabel instead of .ylabel
ax1.legend()


ax2.set_xlabel('Age')
ax2.set_ylabel('Compensation($USD)')
ax2.legend()
plt.show() # Plot subplot


fig, ax = plt.subplots(1,1) # Defualt subplots is 1 row and 1 column

ax.plot(prg.Age,prg.Python,label = 'Python')
ax.plot(prg.Age,prg.JavaScript,label = 'Javascript')
ax.plot(prg.Age,prg.All_Devs,label = 'All Devs')
ax.legend()
ax.set_title('Python and JavaScript Compensation against age')
ax.set_xlabel('Age')
ax.set_ylabel('Compensation($USD)')
# ax.tightlayout() # Adds defualt padding to graph
plt.show()

# WE Can also plot two figures using the following syntax

fig1,ax1 = plt.subplots() # Subplot figure 1
fig2,ax2 = plt.subplots() # Another subplot figure 2

ax1.plot(prg.Age,prg.Python,label = 'Python',color = 'brown')
ax1.set_title('Age against Python compensation')
ax1.set_xlabel('Age')
ax1.set_ylabel('Compensation($USD)')
ax1.legend()

ax2.plot(prg.Age,prg.Python,label = 'Python',color = 'purple')
ax2.set_title('Age against Javascipt compensation')
ax2.set_xlabel('Age')
ax2.set_ylabel('Compensation($USD)')
ax2.legend()

plt.tight_layout()
plt.show()

# Subplot of a lijnear,sqaured and cubic graph and qadraupled

fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex = True)
x = np.array([1,2,3,4,5,6,7,8,9])

ax1.plot(x,x*1,label = 'Linear')
ax1.set_title('Linear Graph')
ax1.set_ylabel('y')
ax1.legend()

ax2.plot(x,x**2,label = 'Quadratic')
ax2.set_title('Quadratic Graph')
ax2.set_ylabel('y')
ax2.legend()

ax3.plot(x,x**3,label = 'Cubic')
ax3.set_title('Cubic Graph')
ax3.set_ylabel('y')
ax3.legend()

ax4.plot(x,x**4,label = 'Quad')
ax4.set_title('Quad Graph')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.legend()

plt.tight_layout()
plt.show()


# Subplots of sin cos 

fig, (ax1,ax2) = plt.subplots(1,2,sharey = True)

x = np.arange(0,2*math.pi,math.pi/180) # Produces numpy array that ranges from 0 to 2pi and step pi/180
swave = np.sin(x) # Takes the sin of each angle in radians
cwave = np.cos(x) # Takes the cos of each sin in radians
twave = np.tan(x)

ax1.plot(x,swave,label = 'sin wave')
ax1.set_title('Sin wave')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()

ax2.plot(x,cwave,label = 'cos wave')
ax2.set_title('Cos Wave')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()

ax3.plot(x,twave,label = 'Tan wave')
ax3.set_title('Tan wave')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.legend()

plt.tight_layout()
plt.show()

# Plotting sin and cos on same graph

plt.plot(x,swave,label = 'Sin Wave',color = 'green')
plt.plot(x,cwave,label = 'Cos Wave',color = 'purple')
plt.title('Sin and Cos Wave')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()

# Time series plots 

stock = pd.read_csv('stock.csv.txt')
print(stock)

# Example first

dates = [
  datetime(2019, 5, 4),
  datetime(2019, 5, 5),
  datetime(2019, 5, 6),
  datetime(2019, 5, 7),
  datetime(2019, 5, 8),
  datetime(2019, 5, 9)
]

y = [0,1,3,4,6,5]

plt.plot_date(dates, y,label = 'stock prices',linestyle = 'dashed') # Default linestyle for plot_date is none so have to specify, Plot data if x or y axis are dates they are set to True and interpreted as dates
plt.title('Stock Prices',fontdict= {'fontweight':'bold'})
plt.ylabel('Dates')
plt.xlabel('Stock Price')
plt.legend()
plt.tight_layout()
# We can manipuate the dates on the figure to rotate them so they are spaced. The current figure has to be called to do this as the plt object is not being used
plt.gcf().autofmt_xdate() # Formats date in current figure 
                       # We can also format the date by changing the order such D/M/Y
plt.show()

x = np.arange(0,5,1)
y = x*2

plt.plot_date(x, y)
# We can manipuate the dates on the figure to rotate them so they are spaced. The current figure has to be called to do this as the plt object is not being used
plt.gcf().autofmt_xdate() # Formats date in current figure 
date_format = mpl_dates.DateFormatter('%A/%m/%Y') # Changes Format of dates
plt.gca().xaxis.set_major_formatter(date_format) # Changes dates on x axis
plt.show()

#Stock Prices

stock = pd.read_csv('stock.csv.txt')
print(stock)
dates_in_stock = pd.to_datetime(stock.Date) # converts dates within csv file into datetime 

plt.plot_date(dates_in_stock, stock.Close,label = 'Close Price',linestyle = 'dashed')
plt.legend()
plt.tight_layout()
plt.title('Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
## Fomratting date to rotate
plt.gcf().autofmt_xdate() # Rotates date a suitable way to avoid conjestion on the x axis
date_format = mpl_dates.DateFormatter('%A/%m/%Y') # Formats date using format strings
plt.gca().xaxis.set_major_formatter(date_format) # Implements formatting into x axis dates
plt.show()

# Covid Line Graphs Time Series 

covid = pd.read_csv('worldwide_covid_data.csv.txt',delimiter = ',')
print(covid)

# We will make 4 subplots 

covid_dates = pd.to_datetime(covid.Date)

fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1)

ax1.plot_date(covid_dates,covid.Confirmed,label = 'Confimed Coronavirus Cases')
ax1.set_title('Coronavirus Cases')
ax1.set_xlabel('Date')
ax1.set_ylabel('Cases')
ax1.legend()

ax2.plot_date(covid_dates,covid.Deaths,label = 'Coronavirus Deaths',linestyle = 'dashed')
ax2.set_title('Coronavirus Deaths',fontdict = {'fontweight':'bold'})
ax2.set_xlabel('Date')
ax2.set_ylabel('Deaths')
ax2.legend()

ax3.plot_date(covid_dates,covid.Recovered,label = 'Coronavirus Recoveries')
ax3.set_title('Coronavirus Recoveries',fontdict = {'fontweight':'bold'})
ax3.set_xlabel('Date')
ax3.set_ylabel('Recoveries')
ax3.legend()

ax4.plot_date(covid.dates,covid['Increase rate'],label = 'Covid rate of increase')
ax4.set_title('Coronavirus rate of increase')
ax4.set_xlabel('Date')
ax4.set_ylabel('Rate')
ax4.legend()

# plt.gcf().autofmt.xdate() # Rotates dates on x axis
plt.show()


























  
























  
  

        
  


  








































