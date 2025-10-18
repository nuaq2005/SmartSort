// SignInScreen.js
import React, { useState } from "react";
import { View, TextInput, Button, Text } from "react-native";
import { auth } from "./firebaseConfig";
import { signInWithEmailAndPassword } from "firebase/auth";

export default function SignInScreen({ navigation }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleSignIn = async () => {
    try {
      await signInWithEmailAndPassword(auth, email, password);
      navigation.navigate("Home"); // go to app home after login
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <View className="flex-1 justify-center p-4 bg-white">
      <Text className="text-xl font-bold mb-4">Sign In</Text>
      <TextInput
        className="border p-2 mb-4 rounded"
        placeholder="Email"
        value={email}
        onChangeText={setEmail}
      />
      <TextInput
        className="border p-2 mb-4 rounded"
        placeholder="Password"
        secureTextEntry
        value={password}
        onChangeText={setPassword}
      />
      {error ? <Text className="text-red-500 mb-2">{error}</Text> : null}
      <Button title="Sign In" onPress={handleSignIn} />
      <Text
        className="mt-4 text-blue-500"
        onPress={() => navigation.navigate("SignUp")}
      >
        Don't have an account? Sign Up
      </Text>
    </View>
  );
}
