import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib
#from wordcloud import WordCloud

# Univariate analysis of continuous variables
class Spam_Detection:
    def Side_By_Side():
        # Compile csv data into dataframe
        df = pd.read_csv("Dataset/whole.csv")

        for column in list(df):
            print("Testing: " + column)
            
            # Set initialization info
            fig = plt.figure(figsize=(20,8),facecolor='white')
            gs = fig.add_gridspec(1,2)
            ax = [None for i in range(2)]
            ax[0] = fig.add_subplot(gs[0,0])
            ax[1] = fig.add_subplot(gs[0,1])
            
            # Set Headings
            ax[0].set_title('Distribution of the ' + column + '\n(Kernel Density Estimate)',fontsize=15,fontweight='bold', fontfamily='monospace')

            # Set Headings
            ax[1].set_title('Distribution of the ' + column + '\n(Histogram Plot)',fontsize=15,fontweight='bold', fontfamily='monospace')

            # Kernel Density Estimate
            sns.kdeplot(x=df[column],ax=ax[0],shade=True, color='gold', alpha=1,zorder=3,linewidth=5,edgecolor='black')
            # The y-axis in a density plot is the probability density function for the kernel density estimation.
            
            # Histogram Plot
            sns.histplot(x=df[column],ax=ax[1], color='olive', alpha=1,zorder=2,linewidth=1,edgecolor='black')

            for i in range(2):
                ax[i].set_ylabel('')
                ax[i].grid(which='both', axis='y', zorder=0, color='black', linestyle=':', dashes=(2,7))
                
                for direction in ['top','right','left']:
                    ax[i].spines[direction].set_visible(False)
                    
                    
            # Rename and save:
            column = column.translate({ord(i): None for i in '!@#$%^&*(){}[],./ ;:\'\"#'})
            plt.savefig("Visualizations/Visualization_of_" + column + ".png")

    def Multiple_Linear_Regression():
        # Reference => https://www.analyticsvidhya.com/blog/2021/05/multiple-linear-regression-using-python-and-scikit-learn/
        from sklearn.linear_model import LinearRegression
        
        x_train, y_train, x_test, y_test = Spam_Detection.import_data()

        # creating an object of LinearRegression class
        LR = LinearRegression()

        # fitting the training data
        LR.fit(x_train,y_train)
        
        # Prediction Classes
        predictions = LR.predict(x_test)
        prediction_classes = [
            1 if prob > 0.5 else 0 for prob in np.ravel(predictions)
        ]
        
        # Pandas Dataframe
        df = pd.DataFrame({'Actual': y_test, 'Predicted': prediction_classes})
        result = df["Actual"].eq(df["Predicted"]).sum() / len(df) * 100
        print(result)
        print(df)

    def Naive_Bayes():
        from sklearn.naive_bayes import BernoulliNB

        x_train, y_train, x_test, y_test = Spam_Detection.import_data()
        
        bnb = BernoulliNB(binarize=0.0)
        bnb.fit(x_train, y_train)

        # Prediction Classes
        predictions = bnb.predict(x_test)
        prediction_classes = [
            1 if prob > 0.5 else 0 for prob in np.ravel(predictions)
        ]
        
        # Pandas Dataframe
        df = pd.DataFrame({'Actual': y_test, 'Predicted': prediction_classes})
        result = df["Actual"].eq(df["Predicted"]).sum() / len(df) * 100
        print(result)
        print(df)
        

        # Visualize
        """
        from sklearn.metrics import confusion_matrix
        mat = confusion_matrix(y_test, labels)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=list(orig_train.columns), yticklabels=train.target_names)
        plt.xlabel('true label')
        plt.ylabel('predicted label');
        """

    def Random_Forest():
        from sklearn.ensemble import RandomForestClassifier

        x_train, y_train, x_test, y_test = Spam_Detection.import_data()
        
        regressor = RandomForestClassifier(n_estimators = 2000, random_state = 55)
        regressor.fit(x_train, y_train) 

        # Prediction Classes
        predictions = regressor.predict(x_test)
        prediction_classes = [
            1 if prob > 0.5 else 0 for prob in np.ravel(predictions)
        ]
        
        # Pandas Dataframe
        df = pd.DataFrame({'Actual': y_test, 'Predicted': prediction_classes})
        result = df["Actual"].eq(df["Predicted"]).sum() / len(df) * 100
        print(result)
        print(df)

        # Visualize
        '''
        data_corr = x_test.corr(method='pearson')
        ax = sns.heatmap(data_corr, vmin=-1, vmax=1, cmap='BrBG')
        
        y_prediction = regressor.predict(x_test)
        x_grid = np.arange(min(x_test), max(x_test), 0.01)
        x_grid = x_grid.reshape((len(x_grid)), 1)

        plt.scatter(x_test, y_test, color="red")
        plt.plot(x_grid, regressor,predict(x_grid), color="blue")
        plt.savefig("Visualizations/Visualization_of_Random_Forest.png")
        '''

    def Gradient_Boosting():
        from sklearn.ensemble import GradientBoostingClassifier

        x_train, y_train, x_test, y_test = Spam_Detection.import_data()
        
        regressor = GradientBoostingClassifier(random_state = 55)
        regressor.fit(x_train, y_train) 

        # Prediction Classes
        predictions = regressor.predict(x_test)
        prediction_classes = [
            1 if prob > 0.5 else 0 for prob in np.ravel(predictions)
        ]
        
        # Pandas Dataframe
        df = pd.DataFrame({'Actual': y_test, 'Predicted': prediction_classes})
        result = df["Actual"].eq(df["Predicted"]).sum() / len(df) * 100
        print(result)
        print(df)

        # Visualize
        '''
        data_corr = x_test.corr(method='pearson')
        ax = sns.heatmap(data_corr, vmin=-1, vmax=1, cmap='BrBG')
        
        y_prediction = regressor.predict(x_test)
        x_grid = np.arange(min(x_test), max(x_test), 0.01)
        x_grid = x_grid.reshape((len(x_grid)), 1)

        plt.scatter(x_test, y_test, color="red")
        plt.plot(x_grid, regressor,predict(x_grid), color="blue")
        plt.savefig("Visualizations/Visualization_of_Random_Forest.png")
        '''

    def SVC_model():
        from sklearn.svm import SVC

        x_train, y_train, x_test, y_test = Spam_Detection.import_data()
        
        regressor = SVC(random_state = 55)
        regressor.fit(x_train, y_train) 

        # Prediction Classes
        predictions = regressor.predict(x_test)
        prediction_classes = [
            1 if prob > 0.5 else 0 for prob in np.ravel(predictions)
        ]
        
        # Pandas Dataframe
        df = pd.DataFrame({'Actual': y_test, 'Predicted': prediction_classes})
        result = df["Actual"].eq(df["Predicted"]).sum() / len(df) * 100
        print(result)
        print(df)

        # Visualize
        '''
        data_corr = x_test.corr(method='pearson')
        ax = sns.heatmap(data_corr, vmin=-1, vmax=1, cmap='BrBG')
        
        y_prediction = regressor.predict(x_test)
        x_grid = np.arange(min(x_test), max(x_test), 0.01)
        x_grid = x_grid.reshape((len(x_grid)), 1)

        plt.scatter(x_test, y_test, color="red")
        plt.plot(x_grid, regressor,predict(x_grid), color="blue")
        plt.savefig("Visualizations/Visualization_of_Random_Forest.png")
        '''

    def Logistic_Model():
        from sklearn.linear_model import LogisticRegression

        x_train, y_train, x_test, y_test = Spam_Detection.import_data()
        
        regressor = LogisticRegression(max_iter=600)
        regressor.fit(x_train, y_train) 

        # Prediction Classes
        predictions = regressor.predict(x_test)
        prediction_classes = [
            1 if prob > 0.5 else 0 for prob in np.ravel(predictions)
        ]
        
        # Pandas Dataframe
        df = pd.DataFrame({'Actual': y_test, 'Predicted': prediction_classes})
        result = df["Actual"].eq(df["Predicted"]).sum() / len(df) * 100
        print(result)
        print(df)

        # Visualize
        '''
        data_corr = x_test.corr(method='pearson')
        ax = sns.heatmap(data_corr, vmin=-1, vmax=1, cmap='BrBG')
        
        y_prediction = regressor.predict(x_test)
        x_grid = np.arange(min(x_test), max(x_test), 0.01)
        x_grid = x_grid.reshape((len(x_grid)), 1)

        plt.scatter(x_test, y_test, color="red")
        plt.plot(x_grid, regressor,predict(x_grid), color="blue")
        plt.savefig("Visualizations/Visualization_of_Random_Forest.png")
        '''

    def Extra_Trees():
        from sklearn.ensemble import ExtraTreesRegressor

        x_train, y_train, x_test, y_test = Spam_Detection.import_data()
        
        regressor = ExtraTreesRegressor()
        regressor.fit(x_train, y_train) 
        
        # Prediction Classes
        predictions = regressor.predict(x_test)
        prediction_classes = [
            1 if prob > 0.5 else 0 for prob in np.ravel(predictions)
        ]
        
        # Pandas Dataframe
        df = pd.DataFrame({'Actual': y_test, 'Predicted': prediction_classes})
        result = df["Actual"].eq(df["Predicted"]).sum() / len(df) * 100
        print(result)
        print(df)

        # Visualize
        '''
        data_corr = x_test.corr(method='pearson')
        ax = sns.heatmap(data_corr, vmin=-1, vmax=1, cmap='BrBG')
        
        y_prediction = regressor.predict(x_test)
        x_grid = np.arange(min(x_test), max(x_test), 0.01)
        x_grid = x_grid.reshape((len(x_grid)), 1)

        plt.scatter(x_test, y_test, color="red")
        plt.plot(x_grid, regressor,predict(x_grid), color="blue")
        plt.savefig("Visualizations/Visualization_of_Random_Forest.png")
        '''

    def Neural_Network():
        import tensorflow as tf
        SHUFFLE_BUFFER = 500
        BATCH_SIZE = 2
        
        x_train, y_train, x_test, y_test = Spam_Detection.import_data()
        
        tf.convert_to_tensor(x_train)

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(x_train)

        #normalizer(x_train)

        def get_basic_model():
            model = tf.keras.Sequential([
                normalizer,
                tf.keras.layers.Dense(10, activation='relu'),
                tf.keras.layers.Dense(10, activation='relu'),
                tf.keras.layers.Dense(1)
            ])

            model.compile(optimizer='adam',
                            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                            metrics=['accuracy'])
            return model


        model = get_basic_model()
        model.fit(x_train, y_train, epochs=1000, batch_size=BATCH_SIZE)

        # model.summary()
        predictions = model.predict(x_test)
        prediction_classes = [
            1 if prob > 0.5 else 0 for prob in np.ravel(predictions)
        ]

        
        df = pd.DataFrame({'Actual': y_test, 'Predicted': prediction_classes})
        result = df["Actual"].eq(df["Predicted"]).sum() / len(df) * 100
        print(result)
        print(df)
        

    def import_data():
        orig_train = pd.read_csv('Dataset/train.csv')
        orig_test = pd.read_csv('Dataset/test.csv')

        # Training
        x_train = orig_train
        x_train = x_train.drop("fake",axis=1)
        x_train = x_train.values
        y_train = orig_train["fake"].values
        
        # Testing
        x_test = orig_test
        x_test = x_test.drop("fake",axis=1)
        x_test = x_test.values
        y_test = orig_test["fake"].values

        return x_train, y_train, x_test, y_test


# Spam_Detection.Side_By_Side()
# Spam_Detection.Multiple_Linear_Regression()
# Spam_Detection.Naive_Bayes()
# Spam_Detection.Random_Forest()
# Spam_Detection.Gradient_Boosting()
# Spam_Detection.SVC_model()
# Spam_Detection.Logistic_Model()
# Spam_Detection.Extra_Trees()
Spam_Detection.Neural_Network()


"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1,2,3])
fig.savefig('test.png')
'''

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)

plt.plot(np.random.randn(50).cumsum(), 'k--')
_ = ax1.hist(np.random.randn(100), bins=20, color='k', alpha=0.3)
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))



#SAVING AND SHOWING 
fig.savefig('test.png')
im = Image.open("test.png")
im.show()
"""

"""
n = range(2,31)

final = 0

for i in n:
    result = ((6*i)-21)
    final += result
    print(result)

print(final)
"""

"""
import numpy as np
from statistics import mode

# reg = [10,15,19,20,22,32,34,37,39,40,45,46]
reg = [4537.3,5660.6, 6428.8, 3029.5, 5785.5, 4853.4, 7045.4, 6965.7, 5040.5, 7266.8, 5043.5, 7257.2, 5716.9, 4937.6, 6590.5]

final = np.mean(reg)
print(final)
final = np.median(reg)
print(final)
final = mode(reg)
print(final)
final = np.std(reg)
final = np.std(reg)
print(final)
"""