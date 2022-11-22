import 'package:flutter/material.dart';

class HomePage extends StatefulWidget {
  const HomePage({Key? key}) : super(key: key);

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: Padding(
          padding: const EdgeInsets.only(left: 23.0),
          child: IconButton(
            icon: Icon(
              Icons.menu,
              color: Colors.grey[800],
              size: 36,
            ),
            onPressed: () {
              //
            },
          ),
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 24.0),
            child: IconButton(
              icon: Icon(
                Icons.person,
                color: Colors.grey[800],
                size: 36,
              ),
              onPressed: () {
                //
              },
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          Text(
            "iwanttoeat",
            style: ,
          ),
        ],
      ),
    );
  }
}
