// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility that Flutter provides. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
class Solution {
public:
class Solution {
public:
    unordered_set<string> vis;
    string s;
    int ans = 1;

    int maxUniqueSplit(string s) {
        this->s = s;
        dfs(0, 0);
        return ans;
    }

    void dfs(int i, int t) {
        if (i >= s.size()) {
            ans = max(ans, t);
            return;
        }
        for (int j = i + 1; j <= s.size(); ++j) {
            string x = s.substr(i, j - i);
            if (!vis.count(x)) {
                vis.insert(x);
                dfs(j, t + 1);
                vis.erase(x);
            }
        }
    }
};
        }
        return ans;
    }
};sNothing);
    expect(find.text('1'), findsOneWidget);
  });
}
