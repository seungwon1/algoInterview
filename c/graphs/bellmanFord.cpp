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

struct Edge{
    int a, b, cost;
};

int n, m, v;
vector<Edge> edges;
const int INF = 1000000000;

[[noreturn]] void solve(int t)
{
    vector<int> d(n, INF);
    vector<int> p(n, -1);
    vector<int> path;
    d[v] = 0;
    int x;
    for (int i = 0; i < n; i++){
        bool any = false;
        x = -1;
        for (Edge e: edges){
            if (d[e.a] < INF){
                d[e.b] = min(d[e.b], d[e.a] + e.cost);
                p[e.b] = e.a;
                any = true;
                x = e.b;
            }
        }
        if (!any){
            break;
        }
    }
    if (x == -1){
        cout << "No negative cycle from " << v;
        if (d[t] == INF){
            cout << "No path from " << v << " to " << t << ".";
        }
        else{
            for (int i = t; i != -1; i = p[i]) {
                path.push_back(i);
            }
            reverse(path.begin(), path.end());
            cout << "Path from " << v << " to " << t << ": ";
            for (int u : path){
                cout << u << ' ';
            }
        }
    } else {
        int y = x;
        for (int i = 0; i < n; ++i){
            y = p[y];
        }
        for (int cur = y;; cur = p[cur]){
            path.push_back(cur);
            if (cur == y && path.size() > 1)
                break;
        }
        reverse(path.begin(), path.end());
    }
    cout << "Negative cycle: ";
    for (int u : path)
        cout << u << ' ';
}