#include <bits/stdc++.h>
using namespace std;
typedef long double lb;


lb error=0;
ofstream training("train.txt");
ofstream testing("test.txt");

void barra(){
    std::cout<<"-------------------------"<<'\n';
}


struct layer{
    int n;
    vector<lb> x;
    vector<vector<lb>> w;
    vector<lb> b;  
    layer()=default;
};


lb h(vector<lb> &x,vector<lb> &w,lb &b){
    lb ans=b;
    for(int i=0;i<x.size();i++) ans=ans+x[i]*w[i];
    return ans;
}


lb activ(lb net, string tipo){
    lb resultado;
    if(tipo == "sigmoide"){
        resultado = 1 / (1 + exp(-net));
    } else if(tipo == "tanh"){
        resultado = tanh(net);
    } else if(tipo == "relu"){
        resultado = net > 0 ? net : 0;
    }
    return resultado;
}


vector<lb> softmax(vector<lb> &vect){
    vector<lb> sol;
    lb suma = 0;
    for(int i=0; i<vect.size(); ++i) suma += exp(vect[i]);
    for(int i=0; i<vect.size(); ++i){
        lb temp = exp(vect[i]) / suma;
        sol.push_back(temp);
    }
    return sol;
}   


void forw(vector<layer> &nn,int &n){
    for(int i=0;i<n-1;i++){
        vector<lb> in_for_next;
        for(int j=0;j<nn[i+1].n;j++){
            if(i!=n-2) in_for_next.push_back(activ(h(nn[i].x,nn[i+1].w[j],nn[i+1].b[j]),"sigmoide"));
            else in_for_next.push_back(h(nn[i].x,nn[i+1].w[j],nn[i+1].b[j]));
        }
        if(i==n-2) nn[i+1].x=softmax(in_for_next);
        else nn[i+1].x=in_for_next;
    }
    return;
}


void back(vector<layer> &nn, vector<lb> &y,int &n, double delta, int samples){
    
    vector<vector<vector<lb>>> derivadas(n);
    
    for(int i=1;i<n;i++){
        vector<vector<lb>> temp_matrix;
        for(int j=0;j<nn[i].n;j++){
            vector<lb> tempp(int(nn[i].w[j].size()),0);
            temp_matrix.push_back(tempp);
        }
        derivadas[i]=temp_matrix;
    }

    for(int s=0;s<samples;s++){

        for(int i=n-1; i>=1; --i){
           
            vector<lb> dp;
            if(i == n-1){
                vector<lb> l1_vect;
                vector<lb> l2_vect;
                // Para la ultima capa
                for(int j=0; j<nn[i].x.size(); ++j){
                    lb l1; //dev respecto costo, bien
                    if(j==y[s]) l1=nn[i].x[j]-1;
                    else l1=nn[i].x[j];
                    lb l2 = nn[i].x[j] * (1 - nn[i].x[j]); // dev respecto act, a(neta)*(1-a(neta)), bien
                    l1_vect.push_back(l1/samples*1.0);
                    l2_vect.push_back(l2);
                }
                
                vector<lb> memoization(nn[i].x.size()); //memoization
                
                for(int j=0; j<l1_vect.size(); ++j){    //dC/dA*dA/dN 
                    memoization[j]=l1_vect[j]*l2_vect[j];
                }

                for(int j=0;j<nn[i].n;j++){ //24
                    for(int k=0;k<nn[i-1].n;k++){ //6
                        lb temp=nn[i-1].x[k]*memoization[j];
                        derivadas[i][j][k]=derivadas[i][j][k]+temp;
                    }
                }

                dp=memoization;
                
            } 
            else{
                vector<lb> l2_vect;
                for(int j=0; j<nn[i].x.size(); ++j){
                    lb l2 = nn[i].x[j] * (1 - nn[i].x[j]); // dev respecto act, a(neta)*(1-a(neta)), bien
                    l2_vect.push_back(l2);
                }
                vector<lb> pre_step;
                for(int j=0; j<1; ++j){
                    for(int z=0; z<nn[i+1].w[0].size(); ++z){
                        lb te;
                        for(int k=0; k<dp.size(); ++k) te = dp[k]*nn[i+1].w[k][j];
                        pre_step.push_back(te);
                    }
                }
                vector<lb> memoization(nn[i].x.size()); //memoization

                for(int j=0; j<l2_vect.size(); ++j) memoization[j]=pre_step[j]*l2_vect[j];
                
                for(int j=0;j<nn[i].n;j++){ 
                    for(int k=0;k<nn[i-1].n;k++){
                        lb temp=nn[i-1].x[k]*memoization[j];
                        derivadas[i][j][k]=derivadas[i][j][k]+temp;
                    }
                }
                dp=memoization;
            }
        }
    }
    //cout<<"lleguee "<<'\n';
    for(int i=1; i<n; ++i){
        for(int j=0; j<derivadas[i].size(); ++j){
                for(int z=0; z<derivadas[i][j].size(); ++z) nn[i].w[j][z] = nn[i].w[j][z] - delta*derivadas[i][j][z];
        }
    }
    return;
}



void initialize(vector<layer> &nn,int &n,int &sz_n){
    for(int i=1;i<n;i++){
        if(i==n-1) nn[i].n=24; 
        else nn[i].n=sz_n;
        for(int j=0;j<nn[i].n;j++){
            int tam=nn[i-1].n;
            vector<lb> w(tam);
            for(int k=0;k<tam;k++) w[k]=(rand()*1.0/RAND_MAX);
            nn[i].w.push_back(w);
            nn[i].b.push_back(1.0*rand()/RAND_MAX);
        }
    }
}


void read_csv(vector<vector<lb>> &X,vector<lb> &Y){
    ifstream read_file("sign_mnist_train.csv");
    string reader;
    while(getline(read_file,reader)){
        string guard="";
        vector<lb> temp;
        bool pri=1;
        for(int i=0;i<reader.size();i++){
            if(reader[i]==','){
                if(pri==1){
                    Y.push_back(stof(guard));
                    pri=0;
                }
                else temp.push_back(stof(guard));
                guard="";
            }
            else guard=guard+reader[i]; 
        }
        temp.push_back(stof(guard));
        X.push_back(temp);
    }
    read_file.close();
}

void read_csv_test(vector<vector<lb>> &X,vector<lb> &Y){
    ifstream read_file("sign_mnist_test.csv");
    string reader;
    while(getline(read_file,reader)){
        string guard="";
        vector<lb> temp;
        bool pri=1;
        for(int i=0;i<reader.size();i++){
            if(reader[i]==','){
                if(pri==1){
                    Y.push_back(stof(guard));
                    pri=0;
                }
                else temp.push_back(stof(guard));
                guard="";
            }
            else guard=guard+reader[i]; 
        }
        temp.push_back(stof(guard));
        X.push_back(temp);
    }
    read_file.close();
}


int main(){

    srand(0);
    vector<vector<lb>> X,X_test;
    vector<lb> Y,Y_test;
    
    read_csv(X,Y);
    read_csv_test(X_test,Y_test);
    
    std::cout<<X.size()<<' '<<X[0].size()<<'\n';
    std::cout<<Y.size()<<'\n';
    
    vector<lb> errores_train;
    vector<lb> errores_test;

    vector<layer> nn(1);//input layer
    nn[0].n=X[0].size();
    
    int n,sz_n;
    std::cout<<"Numero de capas ocultas y de neuronas"<<'\n';
    cin>>n>>sz_n; 
    n=n+2;
    nn.resize(n); //n+2 por la output e input
    initialize(nn,n,sz_n);
    
    for(int i=0;i<n;i++){
        std::cout<<"Capa "<<i<<'\n';
        std::cout<<nn[i].n<<'\n';
        std::cout<<nn[i].x.size()<<'\n';
        std::cout<<nn[i].w.size()<<'\n';
        std::cout<<nn[i].b.size()<<'\n';
        barra();
    }
    
    int iteraciones=100;
    double delta = 0.01;
    
    for(int k=0;k<iteraciones;k++){
        for(int i=0;i<X.size();i++){
            nn[0].x=X[i];
            forw(nn,n);
            for(int j=0;j<24;j++){
                if(j==Y[i]) error=error+(1-nn[n-1].x[j])*(1-nn[n-1].x[j]);
                else error=error+nn[n-1].x[j]*nn[n-1].x[j];
            }
        }
        lb error_test=0;
        for(int i=0;i<X_test.size();i++){
            nn[0].x=X_test[i];
            forw(nn,n);
            for(int j=0;j<24;j++){
                if(j==Y_test[i]) error_test=error_test+(1-nn[n-1].x[j])*(1-nn[n-1].x[j]);
                else error_test=error_test+nn[n-1].x[j]*nn[n-1].x[j];
            }
        }
        
        error=error/(int(X.size())*2);
        error_test=error_test/(int(X_test.size())*2);

        std::cout<<"Error "<<error<<'\n';
        
        errores_train.push_back(error);
        errores_test.push_back(error_test);
        
        //backpropagation
        back(nn, Y,n, delta, X.size());
        error=0;
    }

    for(auto &xyz:errores_train) training<<xyz<<'\n';
    for(auto &xyz:errores_test) testing<<xyz<<'\n';

    int answerpos=0,answerneg=0;

    for(int i=0;i<X_test.size();i++){
            nn[0].x=X_test[i];
            forw(nn,n);
            lb maxx=-1;
            int atrib;
            for(int j=0;j<nn[n-1].n;j++){
                if(maxx<nn[n-1].x[j]) maxx=nn[n-1].x[j],atrib=j;
            }
            if(atrib==Y_test[i]) answerpos++;
            else answerneg++;
    }
    cout<<answerpos<<' '<<answerneg<<'\n';
}
