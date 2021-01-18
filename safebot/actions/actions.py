# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# sys.path.append('C:\\Users\\niles\\OneDrive\\Learn\\rasa\\safebot\\predictions')
# sys.path.append('C:\\Users\\niles\\OneDrive\\Learn\\rasa\\safebot\\actions')
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '\\predictions')
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '\\actions')
# print(sys.path)
from prediction import Predictions,Queries
from actions import *

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormAction
import csv
from rasa_sdk.events import (
    SlotSet,
    UserUtteranceReverted,
    ConversationPaused,
    EventType,
    FollowupAction,
)
import pickle
import pandas as pd
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
import pickle
from sklearn.decomposition import PCA
import os, sys
############################################################################################################################



#############################################################################################################################
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []


class ActionGetTopAccident(Action):

    def name(self) -> Text:
        return "action_get_top_accidents"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
   
        q1=Queries()

        myvar = q1.get_topaccident_description('V')
                
        dispatcher.utter_message(text="{0}".format('\n-------------------------------------------------------------\n'.join(myvar)))

        return []



class ActionGetRiskType(Action):

    def name(self) -> Text:
        return "action_get_risktype"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
   
        myvar = {
        "Unnamed: 0":0,
        "Date" : [datetime.today().strftime('%Y-%m-%d')],
        "Countries": [tracker.get_slot("country")],
        "Local":[tracker.get_slot("location")],
        "Industry": [tracker.get_slot("industry_type")],
        "Potential_Accident":[tracker.get_slot("potential_accident_level")],
        "Gender": [tracker.get_slot("gender")],
        "Emp_Type":  [tracker.get_slot("employee_type")],
        "Critical Risk":[tracker.get_slot("critical_risk")],
        "Description":[tracker.get_slot('zincident_description')]
        # "Description":[tracker.latest_message['text']]
        }

        p1 = Predictions(myvar)
                
        dispatcher.utter_message(text="The accident level based on the data you provided is : {0}".format(p1.predict()))
        # dispatcher.utter_message(text="The predicted accident level is : {0}".format(ctext))

        return []

class UserForm(FormAction):

    def name(self):
        return "user_form"

    @staticmethod
    def required_slots(tracker):
        return ["industry_type","location","country","gender","employee_type","potential_accident_level","zincident_description","critical_risk"]

    
    def submit(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict]:

        # dispatcher.utter_message(template="utter_get_zincident_description")
        dispatcher.utter_message(text = "I have got all the information I needed.I will tell you the accident level ")

        return []


class ActionDefaultAskAffirmation(Action):
   """Asks for an affirmation of the intent if NLU threshold is not met."""

   def name(self):
       return "action_default_ask_affirmation"

   def __init__(self):
        self.intent_mappings = {}
       # read the mapping from a csv and store it in a dictionary
    #    with open('C:/Users/niles/OneDrive/Learn/rasa/safebot/actions/intent_mapping.csv', newline='', encoding='utf-8') as file:
        with open( os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '\\actions\\' + 'intent_mapping.csv', newline='', encoding='utf-8') as file:
           csv_reader = csv.reader(file)
           for row in csv_reader:
               self.intent_mappings[row[0]] = row[1]

   def run(self, dispatcher, tracker, domain):
       # get the most likely intent
       last_intent_name = tracker.get_intent_of_latest_message(skip_fallback_intent =True)
     
       # get the prompt for the intent
    
       intent_prompt = self.intent_mappings[last_intent_name]

       # Create the affirmation message and add two buttons to it.
       # Use '/<intent_name>' as payload to directly trigger '<intent_name>'
       # when the button is clicked.
       message = "Did you mean '{}'?".format(intent_prompt)
       buttons = [{'title': 'Yes',
                   'payload': '/{}'.format(last_intent_name)},
                  {'title': 'No',
                   'payload': '/out_of_scope'}]
       dispatcher.utter_message(message, buttons=buttons)

       return []

class ActionDefaultFallback(Action):
    def name(self) -> Text:
        return "action_default_fallback"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:

        # Fallback caused by TwoStageFallbackPolicy
        if (
            len(tracker.events) >= 4
            and tracker.events[-4].get("name") == "action_default_ask_affirmation"
        ):

            dispatcher.utter_message(template="utter_restart_with_button")

            return [SlotSet("feedback_value", "negative"), ConversationPaused()]

        # Fallback caused by Core
        else:
            dispatcher.utter_message(template="utter_default")
            return [UserUtteranceReverted()]