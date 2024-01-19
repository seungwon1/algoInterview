#include <cmath>
#include <iostream>
#include <tuple>
#include <stdexcept>
#include <string>
#include <cmath>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <list>
#include <stack>
#include <functional>
using namespace std;

const int INF = 1e9;
vector<vector<pair<int, int>>> adj;

void dijkstra(int s, vector<int> & d, vector<int> & p)
{
    int n = adj.size();
    d.assign(n, INF);
    p.assign(n, -1);
    vector<bool> u(n, false);
    for (int i = 0; i < n; i++){
        int v = -1;
        for (int j = 0; j < n; j++){
            if (!u[j] & (v == -1 | d[j] < d[v])){
                v = j;
            }
        }

        if (d[v] == INF){
            break;
        }
        u[v] = true;
        for (auto edge :adj[v]){
            int to = edge.first;
            int w = edge.second;

            if (d[to] > d[v] + w){
                d[to] = d[v] + w;
                p[to] = v;
            }
        }
    }
}

vector<int> restore_path(int s, int t, vector<int> const&p)
{
    vector<int> path;
    for (int v = t; v != s; v = p[v]){
        path.push_back(v);
    }
    path.push_back(s);
    
    reverse(path.begin(), path.end());
    return path;
}


