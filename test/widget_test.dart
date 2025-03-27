class NumberContainers {
    private Map<Integer, Integer> d = new HashMap<>();
    private Map<Integer, TreeSet<Integer>> g = new HashMap<>();class Solution {
    public int maximumSum(int[] nums) {
        int[] d = new int[100];class Solution {class Solution:
    def coloredCells(self, n: int) -> int:class Solution {
public:147.88	147.88class UnionFind {class Solution {
public:
    int minimumIndex(vector<int>& nums) {
        int x = 0, cnt = 0;
        unordered_map<int, int> freq;
        for (int v : nums) {
            ++freq[v];
            if (freq[v] > cnt) {
                cnt = freq[v];
                x = v;
            }
        }
        int cur = 0;
        for (int i = 1; i <= nums.size(); ++i) {
            if (nums[i - 1] == x) {
                ++cur;
                if (cur * 2 > i && (cnt - cur) * 2 > nums.size() - i) {
                    return i - 1;
                }
            }
        }
        return -1;
    }
};
public:
    UnionFind(int n) {
        p = vector<int>(n);class Solution {class Solution {
public:
    int minOperations(vector<vector<int>>& grid, int x) {
        int m = grid.size(), n = grid[0].size();
        int mod = grid[0][0] % x;
        int nums[m * n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] % x != mod) {
                    return -1;
                }
                nums[i * n + j] = grid[i][j];
            }
        }
        sort(nums, nums + m * n);
        int mid = nums[(m * n) >> 1];
        int ans = 0;
        for (int v : nums) {
            ans += abs(v - mid) / x;
        }
        return ans;
    }
};
public:
    int countPaths(int n, vector<vector<int>>& roads) {class Solution {
public:
    int countDays(int days, vector<vector<int>>& meetings) {
        sort(meetings.begin(), meetings.end());
        int ans = 0, last = 0;
        for (auto& e : meetings) {
            int st = e[0], ed = e[1];
            if (last < st) {
                ans += st - last - 1;
            }
            last = max(last, ed);
        }
        ans += days - last;
        return ans;
    }
};
        const long long inf = LLONG_MAX / 2;
        const int mod = 1e9 + 7;

        vector<vector<long long>> g(n, vector<long long>(n, inf));
        for (auto& e : g) {
            fill(e.begin(), e.end(), inf);
        }

        for (auto& r : roads) {
            int u = r[0], v = r[1], t = r[2];
            g[u][v] = t;
            g[v][u] = t;
        }

        g[0][0] = 0;

        vector<long long> dist(n, inf);
        fill(dist.begin(), dist.end(), inf);
        dist[0] = 0;

        vector<long long> f(n);
        f[0] = 1;

        vector<bool> vis(n);
        for (int i = 0; i < n; ++i) {
            int t = -1;
            for (int j = 0; j < n; ++j) {
                if (!vis[j] && (t == -1 || dist[j] < dist[t])) {
                    t = j;
                }
            }
            vis[t] = true;
            for (int j = 0; j < n; ++j) {
                if (j == t) {
                    continue;
                }
                long long ne = dist[t] + g[t][j];
                if (dist[j] > ne) {
                    dist[j] = ne;
                    f[j] = f[t];
                } else if (dist[j] == ne) {
                    f[j] = (f[j] + f[t]) % mod;
                }
            }
        }
        return (int) f[n - 1];
    }
};
        size = vector<int>(n, 1);
        iota(p.begin(), p.end(), 0);
    }

    bool unite(int a, int b) {
        int pa = find(a), pb = find(b);
        if (pa == pb) {
            return false;
        }
        if (size[pa] > size[pb]) {
            p[pb] = pa;
            size[pa] += size[pb];
        } else {
            p[pa] = pb;
            size[pb] += size[pa];
        }
        return true;
    }

    int find(int x) {
        if (p[x] != x) {
            p[x] = find(p[x]);
        }
        return p[x];
    }

    int getSize(int x) {
        return size[find(x)];
    }

private:
    vector<int> p, size;
};

class Solution {
public:
    vector<int> minimumCost(int n, vector<vector<int>>& edges, vector<vector<int>>& query) {
        g = vector<int>(n, -1);
        uf = new UnionFind(n);
        for (auto& e : edges) {
            uf->unite(e[0], e[1]);
        }
        for (auto& e : edges) {
            int root = uf->find(e[0]);
            g[root] &= e[2];
        }
        vector<int> ans;
        for (auto& q : query) {
            ans.push_back(f(q[0], q[1]));
        }
        return ans;
    }

private:
    UnionFind* uf;
    vector<int> g;

    int f(int u, int v) {
        if (u == v) {
            return 0;
        }
        int a = uf->find(u), b = uf->find(v);
        return a == b ? g[a] : -1;
    }
};

    int maximumCount(vector<int>& nums) {
        int a = 0, b = 0;class Solution {
public:
    int minZeroArray(vector<int>& nums, vector<vector<int>>& queries) {
        int n = nums.size();
        int d[n + 1];
        int m = queries.size();
        int l = 0, r = m + 1;
        auto check = [&](int k) -> bool {
            memset(d, 0, sizeof(d));
            for (int i = 0; i < k; ++i) {
                int l = queries[i][0], r = queries[i][1], val = queries[i][2];
                d[l] += val;
                d[r + 1] -= val;
            }
            for (int i = 0, s = 0; i < n; ++i) {
                s += d[i];
                if (nums[i] > s) {
                    return false;
                }
            }
            return true;
        };
        while (l < r) {
            int mid = (l + r) >> 1;
            if (check(mid)) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l > m ? -1 : l;
    }
};
        for (int x : nums) {
            if (x > 0) {
                ++a;
            } else if (x < 0) {
                ++b;
            }
        }
        return max(a, b);
    }
};
class Solution {
public:
    int numberOfSubstrings(string s) {
        int d[3] = {-1, -1, -1};
        int ans = 0;
        for (int i = 0; i < s.size(); ++i) {
            d[s[i] - 'a'] = i;
            ans += min(d[0], min(d[1], d[2])) + 1;
        }
        return ans;
    }
};
                                               
        return 2 * n * (n - 1) + 1class Solution {
public:
    int numberOfAlternatingGroups(vector<int>& colors, int k) {
        int n = colors.size();
        int ans = 0, cnt = 0;
        for (int i = 0; i < n << 1; ++i) {
            if (i && colors[i % n] == colors[(i - 1) % n]) {
                cnt = 1;
            } else {
                ++cnt;
            }
            ans += i >= n && cnt >= k ? 1 : 0;
        }
        return ans;
    }
};
    public int numOfSubarrays(int[] arr) {
        final int mod = (int) 1e9 + 7;class Solution:
    def mergeArrays(
        self, nums1: List[List[int]], nums2: List[List[int]]class Solution:class Solution:class Solution {
public:
    int minimumRecolors(string blocks, int k) {
        int cnt = count(blocks.begin(), blocks.begin() + k, 'W');
        int ans = cnt;
        for (int i = k; i < blocks.size(); ++i) {
            cnt += blocks[i] == 'W';
            cnt -= blocks[i - k] == 'W';
            ans = min(ans, cnt);
        }
        return ans;
    }
};
    def checkPowersOfThree(self, n: int) -> bool:
        while n:
            if n % 3 > 1:
                return False
            n //= 3
        return True
    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        a, b, c = [], [], []
        for x in nums:
            if x < pivot:
                a.append(x)
            elif x == pivot:
                b.append(x)
            else:
                c.append(x)
        return a + b + c
    ) -> List[List[int]]:
        cnt = Counter()
        for i, v in nums1 + nums2:
            cnt[i] += v
        return sorted(cnt.items())
        int[] cnt = {1, 0};class Solution:
    def shortestCommonSupersequence(self, str1: str, str2: str) -> str:
        m, n = len(str1), len(str2)
        f = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):class Solution:
    def applyOperations(self, nums: List[int]) -> List[int]:
        n = len(nums)
        for i in range(n - 1):
            if nums[i] == nums[i + 1]:
                nums[i] <<= 1
                nums[i + 1] = 0
        ans = [0] * n
        i = 0
        for x in nums:
            if x:
                ans[i] = x
                i += 1
        return ans
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    f[i][j] = f[i - 1][j - 1] + 1
                else:
                    f[i][j] = max(f[i - 1][j], f[i][j - 1])
        ans = []
        i, j = m, n
        while i or j:
            if i == 0:
                j -= 1
                ans.append(str2[j])
            elif j == 0:
                i -= 1
                ans.append(str1[i])
            else:
                if f[i][j] == f[i - 1][j]:
                    i -= 1
                    ans.append(str1[i])
                elif f[i][j] == f[i][j - 1]:
                    j -= 1
                    ans.append(str2[j])
                else:
                    i, j = i - 1, j - 1
                    ans.append(str1[i])
        return ''.join(ans[::-1])
        int ans = 0, s = 0;
        for (int x : arr) {
            s += x;
            ans = (ans + cnt[s & 1 ^ 1]) % mod;
            ++cnt[s & 1];
        }class Solution:
    def maxAbsoluteSum(self, nums: List[int]) -> int:
        f = g = 0
        ans = 0
        for x in nums:
            f = max(f, 0) + x
            g = min(g, 0) + x
            ans = max(ans, f, abs(g))
        return ans
        return ans;
    }
}
        int ans = -1;
        for (int v : nums) {
            int x = 0;class Solution {/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* constructFromPrePost(vector<int>& preorder, vector<int>& postorder) {
        unordered_map<int, int> pos;
        int n = postorder.size();
        for (int i = 0; i < n; ++i) {
            pos[postorder[i]] = i;
        }
        function<TreeNode*(int, int, int, int)> dfs = [&](int a, int b, int c, int d) -> TreeNode* {
            if (a > b) {
                return nullptr;
            }
            TreeNode* root = new TreeNode(preorder[a]);
            if (a == b) {
                return root;
            }
            int i = pos[preorder[a + 1]];
            int m = i - c + 1;
            root->left = dfs(a + 1, a + m, c, i);
            root->right = dfs(a + m + 1, b, i + 1, d - 1);
            return root;
        };
        return dfs(0, n - 1, 0, n - 1);
    }
};
    public String findDifferentBinaryString(String[] nums) {
        int mask = 0;
        for (var x : nums) {/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* recoverFromPreorder(string S) {
        stack<TreeNode*> st;
        int depth = 0;
        int num = 0;
        for (int i = 0; i < S.length(); ++i) {
            if (S[i] == '-') {
                depth++;
            } else {
                num = 10 * num + S[i] - '0';
            }
            if (i + 1 >= S.length() || (isdigit(S[i]) && S[i + 1] == '-')) {
                TreeNode* newNode = new TreeNode(num);
                while (st.size() > depth) {
                    st.pop();
                }
                if (!st.empty()) {
                    if (st.top()->left == nullptr) {
                        st.top()->left = newNode;
                    } else {
                        st.top()->right = newNode;
                    }
                }
                st.push(newNode);
                depth = 0;
                num = 0;
            }
        }
        TreeNode* res;
        while (!st.empty()) {
            res = st.top();
            st.pop();
        }
        return res;
    }
};
            int cnt = 0;
            for (int i = 0; i < x.length(); ++i) {
                if (x.charAt(i) == '1') {
                    ++cnt;
                }
            }
            mask |= 1 << cnt;
        }
        for (int i = 0;; ++i) {
            if ((mask >> i & 1) == 0) {
                return "1".repeat(i) + "0".repeat(nums.length - i);
            }
        }
    }
}
            for (int y = v; y > 0; y /= 10) {
                x += y % 10;
            }class Solution {
    private boolean[] vis = new boolean[10];
    private StringBuilder t = new StringBuilder();
    private String p;
    private String ans;

    public String smallestNumber(String pattern) {
        p = pattern;
        dfs(0);
        return ans;
    }

    private void dfs(int u) {
        if (ans != null) {
            return;
        }
        if (u == p.length() + 1) {
            ans = t.toString();
            return;
        }
        for (int i = 1; i < 10; ++i) {
            if (!vis[i]) {
                if (u > 0 && p.charAt(u - 1) == 'I' && t.charAt(u - 1) - '0' >= i) {
                    continue;
                }
                if (u > 0 && p.charAt(u - 1) == 'D' && t.charAt(u - 1) - '0' <= i) {
                    continue;
                }
                vis[i] = true;
                t.append(i);
                dfs(u + 1);
                t.deleteCharAt(t.length() - 1);
                vis[i] = false;
            }
        }
    }
}
            if (d[x] > 0) {
                ans = Math.max(ans, d[x] + v);
            }
            d[x] = Math.max(d[x], v);
        }
        return ans;
    }
}

    public NumberContainers() {
    }
class Solution {
    public long countBadPairs(int[] nums) {
        Map<Integer, Integer> cnt = new HashMap<>();
        long ans = 0;
        for (int i = 0; i < nums.length; ++i) {
            int x = i - nums[i];
            ans += i - cnt.getOrDefault(x, 0);
            cnt.merge(x, 1, Integer::sum);
        }
        return ans;
    }
}
    public void change(int index, int number) {
        if (d.containsKey(index)) {
            int oldNumber = d.get(index);
            g.get(oldNumber).remove(index);
        }
        d.put(index, number);
        g.computeIfAbsent(number, k -> new TreeSet<>()).add(index);
    }

    public int find(int number) {
        var ids = g.get(number);
        return ids == null || ids.isEmpty() ? -1 : ids.first();
    }
}

/**
 * Your NumberContainers object will be instantiated and called as such:
 * NumberContainers obj = new NumberContainers();
 * obj.change(index,number);
 * int param_2 = obj.find(number);
 */class Solution {
public:

        }
        return ans;
    }
};
class Solution {
public:
    long long pickGifts(vector<int>& gifts, int k) {
        make_heap(gifts.begin(), gifts.end());
        while (k--) {
            pop_heap(gifts.begin(), gifts.end());
            gifts.back() = sqrt(gifts.back());
            push_heap(gifts.begin(), gifts.end());
        }
        return accumulate(gifts.begin(), gifts.end(), 0LL);
    }
};
