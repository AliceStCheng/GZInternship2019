import pandas as pd
import matplotlib.pyplot as plt

pd.read_csv('outputs/trained_predct.csv', quoting=2)['output'].hist(bins=30)

plt.title('prediction outputs')
plt.xlabel('prediction percentages')
plt.ylabel('Frequency')

plt.savefig('trained_net_hist.png', dpi=300)
plt.show()
