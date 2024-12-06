class Solution {
public:
    vector<vector<char>> rotateTheBox(vector<vector<char>>& box) {
        int m = box.size(), n = box[0].size();
        vector<vector<char>> ans(n, vector<char>(m));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                ans[j][m - i - 1] = box[i][j];
            }using pii = pair<int, int>;

class Solution {
public:
    bool canChange(string start, string target) {
        auto a = f(start);
        auto b = f(target);
        if (a.size() != b.size()) return false;
        for (int i = 0; i < a.size(); ++i) {
            auto x = a[i], y = b[i];
            if (x.first != y.first) return false;
            if (x.first == 1 && x.second < y.second) return false;
            if (x.first == 2 && x.second > y.second) return false;
        }
        return true;
    }

    vector<pair<int, int>> f(string s) {
        vector<pii> res;
        for (int i = 0; i < s.size(); ++i) {
            if (s[i] == 'L')
                res.push_back({1, i});
            else if (s[i] == 'R')
                res.push_back({2, i});
        }
        return res;
    }
};
class Solution {
public:
    long long maxMatrixSum(vector<vector<int>>& matrix) {
        long long s = 0;
        int cnt = 0, mi = INT_MAX;
        for (auto& row : matrix) {
            for (int& v : row) {
                s += abs(v);
                mi = min(mi, abs(v));
                cnt += v < 0;
            }
        }
        if (cnt % 2 == 0 || mi == 0) return s;
        return s - mi * 2;
    }
};
 class Solution {
public:
    int slidingPuzzle(vector<vector<int>>& board) {
        string start = gets(board);
        string end = "123450";
        if (start == end) return 0;
        unordered_set<string> vis;
        vis.insert(start);
        queue<string> q{{start}};
        int ans = 0;
        while (!q.empty()) {
            ++ans;
            for (int n = q.size(); n > 0; --n) {
                string x = q.front();
                q.pop();
                setb(x, board);
                for (string y : next(board)) {
                    if (y == end) return ans;
                    if (!vis.count(y)) {
                        vis.insert(y);
                        q.push(y);
                    }
                }
            }
        }
        return -1;
    }

    string gets(vector<vector<int>>& board) {
        string s = "";
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j)
                s.push_back('0' + board[i][j]);
        return s;
    }

    void setb(string s, vector<vector<int>>& board) {
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j)
                board[i][j] = s[i * 3 + j] - '0';
    }

    vector<string> next(vector<vector<int>>& board) {
        vector<string> res;
        auto p = find0(board);
        int i = p.first, j = p.second;
        vector<int> dirs = {-1, 0, 1, 0, -1};
        for (int k = 0; k < 4; ++k) {
            int x = i + dirs[k], y = j + dirs[k + 1];
            if (x >= 0 && x < 2 && y >= 0 && y < 3) {
                swap(i, j, x, y, board);
                res.push_back(gets(board));
                swap(i, j, x, y, board);
            }
        }
        return res;
    }

    void swap(int i, int j, int x, int y, vector<vector<int>>& board) {
        int t = board[i][j];
        board[i][j] = board[x][y];
        board[x][y] = t;
    }

    pair<int, int> find0(vector<vector<int>>& board) {
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j)
                if (board[i][j] == 0)
                    return {i, j};
        return {0, 0};
    }
};
class Solution {
public:
    int findChampion(int n, vector<vector<int>>& edges) {
        int indeg[n];
        memset(indeg, 0, sizeof(indeg));
        for (auto& e : edges) {
            ++indeg[e[1]];
        }
        int ans = -1, cnt = 0;
        for (int i = 0; i < n; ++i) {
            if (indeg[i] == 0) {
                ++cnt;
                ans = i;
            }
        }
        return cnt == 1 ? ans : -1;
    }
};
class Solution {
public:
    vector<int> shortestDistanceAfterQueries(int n, vector<vector<int>>& queries) {
        vector<int> g[n];
        for (int i = 0; i < n - 1; ++i) {
            g[i].push_back(i + 1);
        }
        auto bfs = [&](int i) -> int {
            queue<int> q{{i}};
            vector<bool> vis(n);
            vis[i] = true;
            for (int d = 0;; ++d) {
                for (int k = q.size(); k; --k) {
                    int u = q.front();
                    q.pop();
                    if (u == n - 1) {
                        return d;
                    }
                    for (int v : g[u]) {
                        if (!vis[v]) {
                            vis[v] = true;
                            q.push(v);
                        }
                    }
                }
            }
        };
        vector<int> ans;
        for (const auto& q : queries) {
            g[q[0]].push_back(q[1]);
            ans.push_back(bfs(0));
        }
        return ans;
    }
};
class Solution {
public:
    int minimumObstacles(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        deque<tuple<int, int, int>> q{{0, 0, 0}};
        bool vis[m][n];
        memset(vis, 0, sizeof vis);
        int dirs[5] = {-1, 0, 1, 0, -1};
        while (1) {
            auto [i, j, k] = q.front();
            q.pop_front();
            if (i == m - 1 && j == n - 1) {
                return k;
            }
            if (vis[i][j]) {
                continue;
            }
            vis[i][j] = true;
            for (int h = 0; h < 4; ++h) {
                int x = i + dirs[h], y = j + dirs[h + 1];
                if (x >= 0 && x < m && y >= 0 && y < n) {
                    if (grid[x][y] == 0) {
                        q.push_front({x, y, k});
                    } else {
                        q.push_back({x, y, k + 1});
                    }
                }
            }
        }
    }
};
class Solution {
public:
    int minimumTime(vector<vector<int>>& grid) {
        if (grid[0][1] > 1 && grid[1][0] > 1) {
            return -1;
        }
        int m = grid.size(), n = grid[0].size();
        int dist[m][n];
        memset(dist, 0x3f, sizeof dist);
        dist[0][0] = 0;
        using tii = tuple<int, int, int>;
        priority_queue<tii, vector<tii>, greater<tii>> pq;
        pq.emplace(0, 0, 0);
        int dirs[5] = {-1, 0, 1, 0, -1};
        while (1) {
            auto [t, i, j] = pq.top();
            pq.pop();
            if (i == m - 1 && j == n - 1) {
                return t;
            }
            for (int k = 0; k < 4; ++k) {
                int x = i + dirs[k], y = j + dirs[k + 1];
                if (x >= 0 && x < m && y >= 0 && y < n) {
                    int nt = t + 1;
                    if (nt < grid[x][y]) {
                        nt = grid[x][y] + (grid[x][y] - nt) % 2;
                    }
                    if (nt < dist[x][y]) {
                        dist[x][y] = nt;
                        pq.emplace(nt, x, y);
                    }
                }
            }
        }
    }
};
    for (int k = 0; k < 4; ++k) {
                int x = i + dirs[k], y = j + dirs[k + 1];
                if (x >= 0 && x < m && y >= 0 && y < n) {
                    int nt = t + 1;
                    if (nt < grid[x][y]) {
                        nt = grid[x][y] + (grid[x][y] - nt) % 2;
                    }
                    if (nt < dist[x][y]) {
                        dist[x][y] = nt;
                        pq.emplace(nt, x, y);
                    }
                }
            }
        }
    }
};
class Solution {
public:
    bool checkIfExist(vector<int>& arr) {
        unordered_set<int> s;
        for (int x : arr) {
            if (s.contains(x * 2) || (x % 2 == 0 && s.contains(x / 2))) {
                return true;
            }
            s.insert(x);
        }
        return false;
    }
};
class Solution {
public:
    int isPrefixOfWord(string sentence, string searchWord) {
        stringstream ss(sentence);
        string s;
        for (int i = 1; ss >> s; ++i) {
            if (s.find(searchWord) == 0) {
                return i;
            }
        }
        return -1;
    }
};
class Solution {
public:
    string addSpaces(string s, vector<int>& spaces) {
        string ans = "";
        for (int i = 0, j = 0; i < s.size(); ++i) {
            if (j < spaces.size() && i == spaces[j]) {
                ans += ' ';
                ++j;
            }
            ans += s[i];
        }
        return ans;
    }
};
class Solution:
    def maxCount(self, banned: List[int], n: int, maxSum: int) -> int:
        banned.extend([0, n + 1])
        ban = sorted(x for x in set(banned) if x < n + 2)
        ans = 0
        for i, j in pairwise(ban):
            left, right = 0, j - i - 1
            while left < right:
                mid = (left + right + 1) >> 1
                if (i + 1 + i + mid) * mid // 2 <= maxSum:
                    left = mid
                else:
                    right = mid - 1
            ans += left
            maxSum -= (i + 1 + i + left) * left // 2
            if maxSum <= 0:
                break
        return ans
class Solution {
public:
    bool canMakeSubsequence(string str1, string str2) {
        int i = 0, n = str2.size();
        for (char c : str1) {
            char d = c == 'z' ? 'a' : c + 1;
            if (i < n && (str2[i] == c || str2[i] == d)) {
                ++i;
            }
        }
        return i == n;
    }
};
