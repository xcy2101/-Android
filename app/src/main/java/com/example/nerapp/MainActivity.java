package com.example.nerapp;

import android.os.AsyncTask;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    private EditText userInput;
    private Button sendButton;
    private TextView resultTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        userInput = findViewById(R.id.input_edit_text);
        sendButton = findViewById(R.id.submit_button);
        resultTextView = findViewById(R.id.output_text_view);

        sendButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String input = userInput.getText().toString();
                new SendPostRequest().execute(input);
            }
        });
    }

    private class SendPostRequest extends AsyncTask<String, Void, String> {

        @Override
        protected String doInBackground(String... strings) {
            String url = "http://10.0.2.2:5000/process_text";

            OkHttpClient client = new OkHttpClient();

            RequestBody body = RequestBody.create(MediaType.parse("text/plain"), strings[0]);
            Request request = new Request.Builder()
                    .url(url)
                    .post(body)
                    .build();

            try {
                Response response = client.newCall(request).execute();
                return response.body().string();
            } catch (IOException e) {
                e.printStackTrace();
                return "Error: " + e.getMessage();
            }
        }

        @Override
//        回调方法
        protected void onPostExecute(String result) {
            System.out.println(result);
            resultTextView.setText(result);
        }
    }
}
