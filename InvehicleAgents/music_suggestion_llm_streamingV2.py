import os
import json
#from langchain_google_genai import ChatGoogleGenerativeAI
#import google.generativeai as genai
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import spotipy
#from spotipy.oauth2 import SpotifyOAuth 
from spotipy.oauth2 import SpotifyClientCredentials
import response_streamer as res_stream
from langchain_openai import AzureChatOpenAI

# if "GOOGLE_API_KEY" not in os.environ:
#     os.environ["GOOGLE_API_KEY"] = "" 
    
# genai.configure(api_key = os.environ['GOOGLE_API_KEY'])
    
os.environ["OPENAI_API_TYPE"] = "azure_ad"
os.environ["OPENAI_API_VERSION"] = "2024-02-15-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "YOUR_AZURE_OPENAI_ENDPOINT"  # e.g., https://your-resource-name.openai.azure.com/
os.environ["AZURE_OPENAI_API_KEY"] = 'YOUR_AZURE_OPENAI_API_KEY'  # Your Azure OpenAI API key

clientID = 'YOUR_SPOTIFY_CLIENT_ID'       #Spotify API Client ID
clientSecret = 'YOUR_SPOTIFY_CLIENT_SECRET'   #Spotify API Client Secret
redirectURI = 'http://google.com/'
scope = 'streaming'
#auth_manager = SpotifyClientCredentials(client_id=clientID, client_secret=clientSecret, requests_session=True)
#sp = spotipy.Spotify(client_credentials_manager=auth_manager)
oauth_object = spotipy.SpotifyOAuth(clientID,clientSecret,redirectURI,scope=scope)
sp = spotipy.Spotify(client_credentials_manager=oauth_object)
#user = sp.current_user()


class MusicConvo:
    def __init__(self):
        self.memory = ConversationBufferMemory()

    def getSpotifyLink(self, searchQuery, music_type):

        searchResults = sp.search(searchQuery,1,0,music_type[:-1])
        #print(searchQuery)
        #print(music_type[:-1])
        #print(searchResults)
        tracks_dict = searchResults[music_type]
        tracks_items = tracks_dict['items']
        tracklink = tracks_items[0]['external_urls']['spotify']
        return tracklink

    def llm_response_processing(self, str_output):

        try:
            start_str = str_output.find('{')
            end_str = len(str_output)-str_output[::-1].find('}')
            try:
                json_output = json.loads('['+str_output[start_str:end_str]+']')
                #print(str_output[start_str:end_str])
            except:
                json_output = {}
        except:
            start_str = str_output.find('[')
            end_str = str_output.find(']')
            try:
                json_output = json.loads(str_output[start_str:end_str+1])
            except:
                json_output = {}
        
        print('JSON output: ', json_output)
        #json_object = json.dumps(json_output, indent=4)
        
        #Conversational Response
        conv_str = str_output[:str_output.find('\n')]#.replace('\n','')
        #print(conv_str)
        #print(type(json_output))
        music_uri = []   
        
        #if len(json_output) == 0:

        #    conv_str = "Sorry, unable to find your requested track. Please try again with another query."

        #else: 

        for idx, item in enumerate(json_output):

            if item['music_type'] == 'tracks':
                track = item['track_title']
                music_type = 'tracks'
            elif item['music_type'] == 'podcasts':
                track = item['podcast_title']
                if 'latest' or 'recent' in conv_str:
                    music_type = 'shows'
                else:
                    music_type = 'episodes'
            elif item['music_type'] == 'audiobooks':
                track = item['book_title']
                music_type = 'episodes'
            elif item['music_type'] == 'shows':
                track = item['show_title']
                music_type = 'shows'
                
            try:
                artist = item['artist']
            except:
                artist = item['author']

            tracklink = self.getSpotifyLink(track+' '+artist, music_type)
            print(tracklink)
            link_split = tracklink.split('/')
            #print(link_split)
            music_type = link_split[3]
            music_id = link_split[4]
            if len(json_output) > 0:
                conv_str += ' ' + str(idx+1) + '. ' + track + ' by ' + artist
            else:
                conv_str.replace(':','')
            #print(idx+1, track, '-', artist, '-', tracklink, '-', music_type, '-', music_id)
            music_uri.append('spotify:'+music_type+':'+music_id)
            
        return conv_str, music_uri

    def music_llm_call(self, query):
        ##TODO: When giving songs in a list the voice api reads 1. as one pause cause of the periodd
        ## try to remove the period when listing
        
        #llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=1.0, convert_system_message_to_human=True) 
        llm = AzureChatOpenAI(openai_api_version=os.environ["OPENAI_API_VERSION"],azure_deployment="gpt-4o",model_version="2024-05-13")
        
        example_format = """
        [
            {
                "track_title": "abcd",
                "artist": "abcd",
                "music_type": as per the query
            },
            {
                "podcast_title": "efgh",
                "artist": "efgh",
                "music_type": as per the query
            },
            {
                "book_title": "ijkl",
                "author": "ijkl",
                "music_type": as per the query
            },
        ]
        """
        
        PROMPT_TEMPLATE = """
        You are a talkative conversational AI assistant integrated in car's driving system, capable of suggesting music, songs, podcasts, 
        shows and audiobooks based on the driving environment, driver's mood and user's input.
        
        When driver specifies the  input, you will identify the mood, converse and provide your suggestions only in JSON format with 
        keys labelled as 'track_title'/'podcast_title'/'book_title'/'show_title', 'artist'/'author' and 'music_type' depending on type of music.
        Based on your response with title, artist and type, it should be possible to search in Spotify. 
        So, only suggest titles that are definitely present in Spotify.
        Do not include any other keys in the JSON apart from ones mentioned above. 
        
        YOU MUST ALWAYS INCLUDE THE MUSIC TYPE like 'tracks', 'podcasts', 'shows' or 'audiobooks' as another key for each item
        in the JSON. 
               
	Example format: {example_format}               
               
        ALWAYS CONVERSE AND RESPOND TO THE USER'S QUERY WITH A FRIENDLY REPLY AND THEN GIVE YOUR SUGGESTIONS.
        
        LIMIT to a maximum of three suggestions unless stated otherwise by the user.
        Make sure to not use any markdowns while responding.
        
        If requested music not found, mention 'Sorry, unable to find the requested track.' 
        
        Current conversation:
        {history}
        Human: {input}
        AI Assistant:
        """
        
        PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=PROMPT_TEMPLATE, partial_variables={'example_format': example_format})
        
        conversation = ConversationChain(
        llm=llm,
        verbose=False,
        prompt=PROMPT,
        memory=self.memory)
        
        user_input = query
        
        # while user_input.lower() != 'thank you':
            
        str_output = conversation.predict(input=user_input)
        print("raw output of llm",str_output)
        
        stream_response, uri_response = self.llm_response_processing(str_output)
        #print(stream_response)
        #print(uri_response)
        
        #unique_id = '123456789'
        #dict_response = {unique_id:uri_response}
        #stream_response += '###'
        #print(uri_response)

        stream_response = stream_response[2:] if stream_response.startswith("{ ") else stream_response
        #print(stream_response)
        print("uri:" + str(uri_response))
        for word in uri_response:
            stream_response = stream_response + ' #' + word
        
        print("str_resp: " + str(stream_response))
        #print(stream_response)
        #print(dict_response)
        #user_input = query
        return stream_response

        ## Removed to send just the string for logging purposes 
        # async for word in res_stream.words_streamer(stream_response):          
        #         yield f"{word}"

        # else:
        #     stream_response = 'It was a pleasure enhancing your driving experience. Stay safe and enjoy your ride!'
        #     #print(stream_response)

        # async for word in res_stream.words_streamer(stream_response):          
        #     yield f"{word}"
        
