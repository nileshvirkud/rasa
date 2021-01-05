# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

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
class ActionGetRiskType(Action):

    def name(self) -> Text:
        return "action_get_risktype"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        incident_description = tracker.latest_message['text']
   

        data = {
        "Date" : [datetime.today().strftime('%Y-%m-%d')],
        "Countries": [tracker.get_slot("country")],
        "Local":[tracker.get_slot("location")],
        "Industry": [tracker.get_slot("industry_type")],
        "Potential_Accident":[tracker.get_slot("potential_accident_level")],
        "Gender": [tracker.get_slot("gender")],
        "Emp_Type":  [tracker.get_slot("employee_type")],
        "Critical Risk":[tracker.get_slot("critical_risk")],
        "Description":[tracker.latest_message['text']]
        }

        myvar = pd.DataFrame(data)

        # This is incidnet"

        dispatcher.utter_message(text="This is the incident description : {0}".format(myvar['Description']))

        
        # filename = 'predict_risk.pkl'
        # pickle.dump(rf_grid_search, open(filename, 'wb'))
        # loaded_model = pickle.load(open(filename, 'rb'))
        # print(loaded_model)
        # result = loaded_model.predict(X_Test_pca[[1]])
        # print(result) 


        # # get prediction  column cleaned data
        # X_Test =data_val['cleaned_Description'].values
        # y_Test =data_val['Critical Risk'].values
        # #tokenize
        # tokenizer.fit_on_texts(list(X_Test))
        # # to sequences
        # X_Test = tokenizer.texts_to_sequences(X_Test)
        # X_Test = pad_sequences(X_Test, maxlen = maxlen)


        # # scalar
        # X_Test_sc = sc.transform(X_Test)
        # # pca
        # X_Test_pca = pca.transform(X_Test_sc)



        return []

class UserForm(FormAction):

    def name(self):
        return "user_form"

    @staticmethod
    def required_slots(tracker):
        return ["industry_type","location","country","gender","employee_type","potential_accident_level"]

    
    def submit(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict]:

        dispatcher.utter_message(template="utter_get_incident_description")
        return []


class ActionDefaultAskAffirmation(Action):
   """Asks for an affirmation of the intent if NLU threshold is not met."""

   def name(self):
       return "action_default_ask_affirmation"

   def __init__(self):
       self.intent_mappings = {}
       # read the mapping from a csv and store it in a dictionary
       with open('C:/Users/niles/OneDrive/Learn/rasa/safebot/actions/intent_mapping.csv', newline='', encoding='utf-8') as file:
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