import json
import pandas as pd
import matplotlib.pyplot as plt

def main():
  data = []
  with open('beers.json') as f:
    for line in f:
      data.append(json.loads(line))

  df = pd.DataFrame(data)
# print df[dataFr['brewery'].str.contains('Russian') == True]
# print df.columns


# print df['style'].value_counts(normalize=True)
# print df['style'].value_counts(normalize=False)
# sorted_df = df.sort(['style'])
# print df['style'].describe()
# print sorted_df['style']

# Make sure none of the styles contain the same names (IPA for example appears in both the first and second searches but ' IPA' does not) then sum the number in the category
  print df['style'][df['style'].str.contains('Pale Ale') == True].unique()
  print df['style'][df['style'].str.contains(' IPA') == True].unique()

  Pale = df['style'][df['style'].str.contains('Pale Ale') == True].value_counts().sum() \
       + df['style'][df['style'].str.contains(' IPA') == True].value_counts().sum()
  print

# Do the same for Lagers
  print df['style'][df['style'].str.contains('Lager') == True].unique()
  Lager = df['style'][df['style'].str.contains('Lager') == True].value_counts().sum()
  print 

# Do the same for stouts and porters  
  print df['style'][df['style'].str.contains('Porter') == True].unique()
  print df['style'][df['style'].str.contains('Stout') == True].unique()
  print df['style'][df['style'].str.contains('Brown') == True].unique()

  S_P = df['style'][df['style'].str.contains('Porter') == True].value_counts().sum() \
      + df['style'][df['style'].str.contains('Stout') == True].value_counts().sum() \
      + df['style'][df['style'].str.contains('Brown') == True].value_counts().sum()
  print

#Do the same for sours and wilds
  print df['style'][df['style'].str.contains('Lambic') == True].unique()
  print df['style'][df['style'].str.contains('Gueuze') == True].unique()
  print df['style'][df['style'].str.contains('Flanders') == True].unique()
  print df['style'][df['style'].str.contains('Gose') == True].unique()
  print df['style'][df['style'].str.contains('Wild') == True].unique()
  print df['style'][df['style'].str.contains('Saison') == True].unique()
  print df['style'][df['style'].str.contains('Berliner') == True].unique()
  Wild = df['style'][df['style'].str.contains('Lambic') == True].value_counts().sum() \
       + df['style'][df['style'].str.contains('Gueuze') == True].value_counts().sum() \
       + df['style'][df['style'].str.contains('Flanders') == True].value_counts().sum() \
       + df['style'][df['style'].str.contains('Gose') == True].value_counts().sum() \
       + df['style'][df['style'].str.contains('Wild') == True].value_counts().sum() \
       + df['style'][df['style'].str.contains('Saison') == True].value_counts().sum() \
       + df['style'][df['style'].str.contains('Berliner') == True].value_counts().sum()
  print

  Sub_Total = float(sum([Pale, Lager, S_P, Wild]))
# print df['style'].count(), Total
  Total = float(df['style'].count())
  Percentages = [float(Pale)/Total, float(Lager)/Total, float(S_P)/Total, float(Wild)/Total, (Total-Sub_Total)/Total]

# Pie chart of the groups listed above
  labels = 'Pale Ales', 'Lagers', 'Stouts/Porters', 'Wild Ales', 'Other'
  explode = (0, 0, 0, 0.1, 0.0) # only "explode" the 4th slice (Wild Ales)

  plt.pie(Percentages, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

# Set aspect ratio to be equal so that pie is drawn as a circle.
  plt.axis('equal')

  plt.savefig('Beer_pie.png', bbox_inches='tight')
  plt.show()

# Bar chart of the different categories in Wild Ales
  labels = ['Lambic', 'Gueuze', 'Flanders', 'Gose', 'Wild', 'Saison', 'Berliner Weissbier']
  plt.bar(range(7) \
       , [df['style'][df['style'].str.contains('Lambic') == True].value_counts().sum() \
       , df['style'][df['style'].str.contains('Gueuze') == True].value_counts().sum() \
       , df['style'][df['style'].str.contains('Flanders') == True].value_counts().sum() \
       , df['style'][df['style'].str.contains('Gose') == True].value_counts().sum() \
       , df['style'][df['style'].str.contains('Wild') == True].value_counts().sum() \
       , df['style'][df['style'].str.contains('Saison') == True].value_counts().sum() \
       , df['style'][df['style'].str.contains('Berliner') == True].value_counts().sum()] \
       , align='center')
  plt.xticks(range(7), labels, rotation=45)
# plt.xlabel('Beer Style')
  plt.ylabel('Number of Beers Within Style')

  plt.savefig('Wild_bar.png', bbox_inches='tight')
  plt.show()

# Pie chart of all the individual styles
# plt.pie(df['style'].value_counts(normalize=True), autopct='%1.1f%%', shadow=True, startangle=90)

# plt.pie(, explode=explode, labels=labels, colors=colors,
#         autopct='%1.1f%%', shadow=True, startangle=90)
# Set aspect ratio to be equal so that pie is drawn as a circle.
# plt.axis('equal')

# plt.show()

if __name__ == '__main__':
  main()
