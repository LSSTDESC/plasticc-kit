import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
mpl.rcParams['font.size'] = 15.0


labels = ['Astronomy', 'Other STEM', 'Not Software',  'Unknown', 'Software/Data Science']
sizes = [51, 90, 65, 666, 443]
#colors
colors = ['#1F0322', '#8A1C7C', '#DA4167', '#F0BCD4', '#899D78']

#ff9999','#66b3ff','#99ff99','#ffcc99', 'cyan']
#explsion
explode = (0.05,0.05,0.05,0.05,0.02)
fig1, ax1 = plt.subplots(figsize=(15,12))
 
ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%',startangle=90, pctdistance=0.85, explode = explode)
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.savefig('pie.png')
