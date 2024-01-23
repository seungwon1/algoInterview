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
#include <set>
using namespace std;


const int INF = 1e9;
vector<vector<pair<int, int>>> g;


void bfs01(int s){
    int n = g.size();
    vector<int> d(n, INF);
    d[s] = 0;
    deque<int> q;
    q.push_front(s);
    while (!q.empty()){
        int v = q.front();
        q.pop_front();
        for (auto edge: g[v]){
            int to = edge.first;
            int w = edge.second;
            if (d[to] > w + d[v]){
                if (w == 0){
                    q.push_front(to);
                } else {
                    q.push_back(to);
                }
            }
        }
    }
}