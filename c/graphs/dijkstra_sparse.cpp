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
#include <queue>

using namespace std;

const int INF = 1e9;
vector<vector<pair<int, int>>> adj;

void dijkstraSet(int s, vector<int> &d, vector<int> &p)
{
    int n = adj.size();
    d.assign(n, INF);
    p.assign(-1, INF);

    d[s] = 0;
    set<pair<int, int>> q;
    q.insert({0, s});
    while (!q.empty()){
        int v = q.begin()->second;
        q.erase(q.begin());

        for (auto edge: adj[v]){
            int to = edge.first;
            int w = edge.second;

            if (d[v] + w < d[to]){
                d[to] = d[v] + w;
                p[to] = v;
                q.insert({to, d[to]});
            }
        }
    }
}

void dijkstraQue(int s, vector<int> &d, vector<int> &p)
{
    int n = adj.size();
    d.assign(n, INF);
    p.assign(-1, INF);

    d[s] = 0;
    using pii = pair<int, int>;
    priority_queue<pii, vector<pii>, greater<pii>> q;
    q.push({0, s});
    while (!q.empty()) {
        int v = q.top().second;
        int d_v = q.top().first;
        q.pop();
        if (d_v != d[v]) {
            continue;
        }
        for (auto edge: adj[v]) {
            int to = edge.first;
            int w = edge.second;

            if (d[v] + w < d[to]) {
                d[to] = d[v] + w;
                p[to] = v;
                q.push({to, d[to]});
            }
        }
    }
}