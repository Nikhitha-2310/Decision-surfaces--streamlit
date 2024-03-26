
import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import mlxtend
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


ushape=pd.read_csv(r'C:\Users\LENOVO\Decision_srufaces\Multiple CSV\1.ushape.csv',header=None,names=['f1','f2','cv'])
concentriccir1=pd.read_csv(r'C:\Users\LENOVO\Decision_srufaces\Multiple CSV\2.concerticcir1.csv',header=None,names=['f1','f2','cv'])
concentriccir2=pd.read_csv(r'C:\Users\LENOVO\Decision_srufaces\Multiple CSV\3.concertriccir2.csv',header=None,names=['f1','f2','cv'])
linearsep=pd.read_csv(r'C:\Users\LENOVO\Decision_srufaces\Multiple CSV\4.linearsep.csv',header=None,names=['f1','f2','cv'])
outlier=pd.read_csv(r'C:\Users\LENOVO\Decision_srufaces\Multiple CSV\5.outlier.csv',header=None,names=['f1','f2','cv'])
overlap=pd.read_csv(r'C:\Users\LENOVO\Decision_srufaces\Multiple CSV\6.overlap.csv',header=None,names=['f1','f2','cv'])
xor=pd.read_csv(r'C:\Users\LENOVO\Decision_srufaces\Multiple CSV\7.xor.csv',header=None,names=['f1','f2','cv'])
twospirals=pd.read_csv(r'C:\Users\LENOVO\Decision_srufaces\Multiple CSV\8.twospirals.csv',header=None,names=['f1','f2','cv'])
random=pd.read_csv(r'C:\Users\LENOVO\Decision_srufaces\Multiple CSV\9.random.csv',header=None,names=['f1','f2','cv'])


st.title('Decision Surfaces')

with st.sidebar:
    st.title('ExploreZone')
    radio_button=st.radio('Datasets',['ushape','concentriccir1','concentriccir2','linearsep','outlier','overlap','xor','twospirals','random'])
    k=st.slider('Kvalue',min_value=1,max_value=15)
    radio_button2=st.radio('Dec_surfaces',['Sinlge','Multiple'])
    

if(radio_button=='ushape'):
    st.write('Ushape')
    st.scatter_chart(data=ushape,x='f1',y='f2')
    fv=ushape.iloc[:,:2]
    cv=ushape.iloc[:,-1]
    std=StandardScaler()
    p_fv=std.fit_transform(fv)
    if(radio_button2=='Sinlge'):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(p_fv,cv.astype(int))
        fig, ax = plt.subplots()
        plot_decision_regions(X=p_fv, y=cv.astype(int).values, clf=knn, ax=ax)
        st.pyplot(fig)
        st.write('{}NN'.format(k))
    elif(radio_button2=='Multiple'):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for i, ax in zip(range(1, k, 2), axes.flatten()):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(p_fv, cv.astype(int))
            plot_decision_regions(X=p_fv, y=cv.astype(int).values, clf=knn, ax=ax)
            ax.set_title(f'KNN with n_neighbors={i}')

        plt.tight_layout()
        st.pyplot(fig)

    x_train,x_test,y_train,y_test=train_test_split(fv,cv,test_size=0.2,random_state=1,stratify=cv)
    std=StandardScaler()
    px_train=std.fit_transform(x_train)
    px_test=std.transform(x_test)
    

    train_error=[]
    test_error=[]
    error={}
    for i in range(1,k,2):
        knn=KNeighborsClassifier(n_neighbors=i)
        model=knn.fit(px_train,y_train)
        predicted_train=model.predict(px_train)
        train_error.append(1-accuracy_score(y_train,predicted_train))
        predicted_test=model.predict(px_test)
        test_error.append(1-accuracy_score(y_test,predicted_test))
    error = {'Train Error': train_error, 'Test Error': test_error}
    error_df = pd.DataFrame(error)
    st.line_chart(error_df)

if(radio_button=='concentriccir1'):
    st.scatter_chart(data=concentriccir1,x='f1',y='f2')
    st.write('concentriccir1')
    fv=concentriccir1.iloc[:,:2]
    cv=concentriccir1.iloc[:,-1]
    std=StandardScaler()
    p_fv=std.fit_transform(fv)
    if(radio_button2=='Sinlge'):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(p_fv,cv.astype(int))
        fig, ax = plt.subplots()
        plot_decision_regions(X=p_fv, y=cv.astype(int).values, clf=knn, ax=ax)
        st.pyplot(fig)
        st.write('{}NN'.format(k))
    elif(radio_button2=='Multiple'):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for i, ax in zip(range(1, k, 2), axes.flatten()):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(p_fv, cv.astype(int))
            plot_decision_regions(X=p_fv, y=cv.astype(int).values, clf=knn, ax=ax)
            ax.set_title(f'KNN with n_neighbors={i}')

        plt.tight_layout()
        st.pyplot(fig)

    x_train,x_test,y_train,y_test=train_test_split(fv,cv,test_size=0.2,random_state=1,stratify=cv)
    std=StandardScaler()
    px_train=std.fit_transform(x_train)
    px_test=std.transform(x_test)
    

    train_error=[]
    test_error=[]
    error={}
    for i in range(1,k,2):
        knn=KNeighborsClassifier(n_neighbors=i)
        model=knn.fit(px_train,y_train)
        predicted_train=model.predict(px_train)
        train_error.append(1-accuracy_score(y_train,predicted_train))
        predicted_test=model.predict(px_test)
        test_error.append(1-accuracy_score(y_test,predicted_test))
    error = {'Train Error': train_error, 'Test Error': test_error}
    error_df = pd.DataFrame(error)
    st.line_chart(error_df)

if(radio_button=='concentriccir2'):
    st.scatter_chart(data=concentriccir2,x='f1',y='f2')
    st.write('concentriccir2')
    fv=concentriccir2.iloc[:,:2]
    cv=concentriccir2.iloc[:,-1]
    std=StandardScaler()
    p_fv=std.fit_transform(fv)
    if(radio_button2=='Sinlge'):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(p_fv,cv.astype(int))
        fig, ax = plt.subplots()
        plot_decision_regions(X=p_fv, y=cv.astype(int).values, clf=knn, ax=ax)
        st.pyplot(fig)
        st.write('{}NN'.format(k))
    elif(radio_button2=='Multiple'):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for i, ax in zip(range(1, k, 2), axes.flatten()):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(p_fv, cv.astype(int))
            plot_decision_regions(X=p_fv, y=cv.astype(int).values, clf=knn, ax=ax)
            ax.set_title(f'KNN with n_neighbors={i}')

        plt.tight_layout()
        st.pyplot(fig)

    x_train,x_test,y_train,y_test=train_test_split(fv,cv,test_size=0.2,random_state=1,stratify=cv)
    std=StandardScaler()
    px_train=std.fit_transform(x_train)
    px_test=std.transform(x_test)
    

    train_error=[]
    test_error=[]
    error={}
    for i in range(1,k,2):
        knn=KNeighborsClassifier(n_neighbors=i)
        model=knn.fit(px_train,y_train)
        predicted_train=model.predict(px_train)
        train_error.append(1-accuracy_score(y_train,predicted_train))
        predicted_test=model.predict(px_test)
        test_error.append(1-accuracy_score(y_test,predicted_test))
    error = {'Train Error': train_error, 'Test Error': test_error}
    error_df = pd.DataFrame(error)
    st.line_chart(error_df)

if(radio_button=='linearsep'):
    st.scatter_chart(data=linearsep,x='f1',y='f2')
    st.write('linearsep')
    fv=linearsep.iloc[:,:2]
    cv=linearsep.iloc[:,-1]
    std=StandardScaler()
    p_fv=std.fit_transform(fv)
    if(radio_button2=='Sinlge'):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(p_fv,cv.astype(int))
        fig, ax = plt.subplots()
        plot_decision_regions(X=p_fv, y=cv.astype(int).values, clf=knn, ax=ax)
        st.pyplot(fig)
        st.write('{}NN'.format(k))
    elif(radio_button2=='Multiple'):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for i, ax in zip(range(1, k, 2), axes.flatten()):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(p_fv, cv.astype(int))
            plot_decision_regions(X=p_fv, y=cv.astype(int).values, clf=knn, ax=ax)
            ax.set_title(f'KNN with n_neighbors={i}')

        plt.tight_layout()
        st.pyplot(fig)

    x_train,x_test,y_train,y_test=train_test_split(fv,cv,test_size=0.2,random_state=1,stratify=cv)
    std=StandardScaler()
    px_train=std.fit_transform(x_train)
    px_test=std.transform(x_test)
    

    train_error=[]
    test_error=[]
    error={}
    for i in range(1,k,2):
        knn=KNeighborsClassifier(n_neighbors=i)
        model=knn.fit(px_train,y_train)
        predicted_train=model.predict(px_train)
        train_error.append(1-accuracy_score(y_train,predicted_train))
        predicted_test=model.predict(px_test)
        test_error.append(1-accuracy_score(y_test,predicted_test))
    error = {'Train Error': train_error, 'Test Error': test_error}
    error_df = pd.DataFrame(error)
    st.line_chart(error_df)

if(radio_button=='outlier'):
    st.scatter_chart(data=outlier,x='f1',y='f2')
    st.write('outlier')
    fv=outlier.iloc[:,:2]
    cv=outlier.iloc[:,-1]
    std=StandardScaler()
    p_fv=std.fit_transform(fv)
    if(radio_button2=='Sinlge'):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(p_fv,cv.astype(int))
        fig, ax = plt.subplots()
        plot_decision_regions(X=p_fv, y=cv.astype(int).values, clf=knn, ax=ax)
        st.pyplot(fig)
        st.write('{}NN'.format(k))
    elif(radio_button2=='Multiple'):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for i, ax in zip(range(1, k, 2), axes.flatten()):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(p_fv, cv.astype(int))
            plot_decision_regions(X=p_fv, y=cv.astype(int).values, clf=knn, ax=ax)
            ax.set_title(f'KNN with n_neighbors={i}')

        plt.tight_layout()
        st.pyplot(fig)

    x_train,x_test,y_train,y_test=train_test_split(fv,cv,test_size=0.2,random_state=1,stratify=cv)
    std=StandardScaler()
    px_train=std.fit_transform(x_train)
    px_test=std.transform(x_test)
    

    train_error=[]
    test_error=[]
    error={}
    for i in range(1,k,2):
        knn=KNeighborsClassifier(n_neighbors=i)
        model=knn.fit(px_train,y_train)
        predicted_train=model.predict(px_train)
        train_error.append(1-accuracy_score(y_train,predicted_train))
        predicted_test=model.predict(px_test)
        test_error.append(1-accuracy_score(y_test,predicted_test))
    error = {'Train Error': train_error, 'Test Error': test_error}
    error_df = pd.DataFrame(error)
    st.line_chart(error_df)

if(radio_button=='overlap'):
    st.scatter_chart(data=overlap,x='f1',y='f2')
    st.write('overlap')
    fv=overlap.iloc[:,:2]
    cv=overlap.iloc[:,-1]
    std=StandardScaler()
    p_fv=std.fit_transform(fv)
    if(radio_button2=='Sinlge'):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(p_fv,cv.astype(int))
        fig, ax = plt.subplots()
        plot_decision_regions(X=p_fv, y=cv.astype(int).values, clf=knn, ax=ax)
        st.pyplot(fig)
        st.write('{}NN'.format(k))
    elif(radio_button2=='Multiple'):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for i, ax in zip(range(1, k, 2), axes.flatten()):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(p_fv, cv.astype(int))
            plot_decision_regions(X=p_fv, y=cv.astype(int).values, clf=knn, ax=ax)
            ax.set_title(f'KNN with n_neighbors={i}')

        plt.tight_layout()
        st.pyplot(fig)

    x_train,x_test,y_train,y_test=train_test_split(fv,cv,test_size=0.2,random_state=1,stratify=cv)
    std=StandardScaler()
    px_train=std.fit_transform(x_train)
    px_test=std.transform(x_test)
    

    train_error=[]
    test_error=[]
    error={}
    for i in range(1,k,2):
        knn=KNeighborsClassifier(n_neighbors=i)
        model=knn.fit(px_train,y_train)
        predicted_train=model.predict(px_train)
        train_error.append(1-accuracy_score(y_train,predicted_train))
        predicted_test=model.predict(px_test)
        test_error.append(1-accuracy_score(y_test,predicted_test))
    error = {'Train Error': train_error, 'Test Error': test_error}
    error_df = pd.DataFrame(error)
    st.line_chart(error_df)

if(radio_button=='xor'):
    st.scatter_chart(data=xor,x='f1',y='f2')
    st.write('xor')
    fv=xor.iloc[:,:2]
    cv=xor.iloc[:,-1]
    std=StandardScaler()
    p_fv=std.fit_transform(fv)
    if(radio_button2=='Sinlge'):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(p_fv,cv.astype(int))
        fig, ax = plt.subplots()
        plot_decision_regions(X=p_fv, y=cv.astype(int).values, clf=knn, ax=ax)
        st.pyplot(fig)
        st.write('{}NN'.format(k))
    elif(radio_button2=='Multiple'):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for i, ax in zip(range(1, k, 2), axes.flatten()):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(p_fv, cv.astype(int))
            plot_decision_regions(X=p_fv, y=cv.astype(int).values, clf=knn, ax=ax)
            ax.set_title(f'KNN with n_neighbors={i}')

        plt.tight_layout()
        st.pyplot(fig)

    x_train,x_test,y_train,y_test=train_test_split(fv,cv,test_size=0.2,random_state=1,stratify=cv)
    std=StandardScaler()
    px_train=std.fit_transform(x_train)
    px_test=std.transform(x_test)
    

    train_error=[]
    test_error=[]
    error={}
    for i in range(1,k,2):
        knn=KNeighborsClassifier(n_neighbors=i)
        model=knn.fit(px_train,y_train)
        predicted_train=model.predict(px_train)
        train_error.append(1-accuracy_score(y_train,predicted_train))
        predicted_test=model.predict(px_test)
        test_error.append(1-accuracy_score(y_test,predicted_test))
    error = {'Train Error': train_error, 'Test Error': test_error}
    error_df = pd.DataFrame(error)
    st.line_chart(error_df)

if(radio_button=='twospirals'):
    st.scatter_chart(data=twospirals,x='f1',y='f2')
    st.write('twospirals')
    fv=twospirals.iloc[:,:2]
    cv=twospirals.iloc[:,-1]
    std=StandardScaler()
    p_fv=std.fit_transform(fv)
    if(radio_button2=='Sinlge'):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(p_fv,cv.astype(int))
        fig, ax = plt.subplots()
        plot_decision_regions(X=p_fv, y=cv.astype(int).values, clf=knn, ax=ax)
        st.pyplot(fig)
        st.write('{}NN'.format(k))
    elif(radio_button2=='Multiple'):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for i, ax in zip(range(1, k, 2), axes.flatten()):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(p_fv, cv.astype(int))
            plot_decision_regions(X=p_fv, y=cv.astype(int).values, clf=knn, ax=ax)
            ax.set_title(f'KNN with n_neighbors={i}')

        plt.tight_layout()
        st.pyplot(fig)

    x_train,x_test,y_train,y_test=train_test_split(fv,cv,test_size=0.2,random_state=1,stratify=cv)
    std=StandardScaler()
    px_train=std.fit_transform(x_train)
    px_test=std.transform(x_test)
    

    train_error=[]
    test_error=[]
    error={}
    for i in range(1,k,2):
        knn=KNeighborsClassifier(n_neighbors=i)
        model=knn.fit(px_train,y_train)
        predicted_train=model.predict(px_train)
        train_error.append(1-accuracy_score(y_train,predicted_train))
        predicted_test=model.predict(px_test)
        test_error.append(1-accuracy_score(y_test,predicted_test))
    error = {'Train Error': train_error, 'Test Error': test_error}
    error_df = pd.DataFrame(error)
    st.line_chart(error_df)
    

if(radio_button=='random'):
    st.scatter_chart(data=random,x='f1',y='f2')
    st.write('random')
    fv=random.iloc[:,:2]
    cv=random.iloc[:,-1]
    std=StandardScaler()
    p_fv=std.fit_transform(fv)
    if(radio_button2=='Sinlge'):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(p_fv,cv.astype(int))
        fig, ax = plt.subplots()
        plot_decision_regions(X=p_fv, y=cv.astype(int).values, clf=knn, ax=ax)
        st.pyplot(fig)
        st.write('{}NN'.format(k))
    elif(radio_button2=='Multiple'):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for i, ax in zip(range(1, k, 2), axes.flatten()):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(p_fv, cv.astype(int))
            plot_decision_regions(X=p_fv, y=cv.astype(int).values, clf=knn, ax=ax)
            ax.set_title(f'KNN with n_neighbors={i}')

        plt.tight_layout()
        st.pyplot(fig)

    x_train,x_test,y_train,y_test=train_test_split(fv,cv,test_size=0.2,random_state=1,stratify=cv)
    std=StandardScaler()
    px_train=std.fit_transform(x_train)
    px_test=std.transform(x_test)
    

    train_error=[]
    test_error=[]
    error={}
    for i in range(1,k,2):
        knn=KNeighborsClassifier(n_neighbors=i)
        model=knn.fit(px_train,y_train)
        predicted_train=model.predict(px_train)
        train_error.append(1-accuracy_score(y_train,predicted_train))
        predicted_test=model.predict(px_test)
        test_error.append(1-accuracy_score(y_test,predicted_test))
    error = {'Train Error': train_error, 'Test Error': test_error}
    error_df = pd.DataFrame(error)
    st.line_chart(error_df)


