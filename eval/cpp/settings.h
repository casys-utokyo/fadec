#pragma once

#include <bits/stdc++.h>
using namespace std;

#define setwp setw(12) << setprecision(5)

#define FOR(i, a, b) for (int i = (a); i < (b); i++)
#define REP(i, n) for(int i = 0; i < (n); i++)
#define printyn(b) cout << ((b) ? "Yes\n" : "No\n")
#define delim(i, n) cout << (i == (n)-1 ? "\n" : " ")
#define print1(v) cout << setwp << (v) << "\n"
#define print2(v1, v2) cout << setwp << (v1) << " " << (v2) << "\n"
#define print3(v1, v2, v3) cout << setwp << (v1) << " " << (v2) << " " << (v3) << "\n"
#define print4(v1, v2, v3, v4) cout << setwp << (v1) << " " << (v2) << " " << (v3) << " " << (v4) << "\n"
#define printvi(v, n) REP(iiii, n) { cout << setwp << v[iiii]; delim(iiii, n); }
#define printvii(v, h, w) REP(iiiii, h) { printvi(v[iiiii], w); }
#define printviii(v, d, h, w) REP(iiiiii, d) { printvii(v[iiiiii], h, w); cout << "\n"; }
#define printviiii(v, c, d, h, w) REP(iiiiiii, c) { printviii(v[iiiiiii], d, h, w); cout << "------------------\n\n"; }
#define printP(p) cout << setwp << "(" << (p).first << ", " << (p).second << ")\n"
