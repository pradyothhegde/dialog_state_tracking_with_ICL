# Pradyoth Hegde (pradyothhegde@gmail.com), Santosh Kesiraju (kesiraju@fit.vutbr.cz)
# Reconciling the multiwoz2.4 data.json with the goal and log domains.

import json
from tqdm import tqdm

input_data_path = '/mnt/matylda4/hegde/int_ent/LLM_dialog_state/Data/MultiWOZ_2.1/data.json'
output_data_path = '/mnt/matylda4/hegde/int_ent/LLM_dialog_state/Data/MultiWOZ_2.1/data_refined.json'

valid_domain_list = ["taxi", "police", "hospital", "hotel", "attraction", "train", "restaurant", "bus"] # also there is bus!!but not in conversation!
valid_fname_domain_pair_dict = {}
with open(input_data_path) as in_file:
    data = json.load(in_file)

for key, value in tqdm(data.items()):
    
    goal = data.get(key).get("goal")
    log = data.get(key).get("log")
    # print(key)
    # print(value)

    ### goal operation
    valid_fname_domain_pair_dict.update(fname=key)
    temp_domain_list = []
    for goal_domain_and_others in goal.keys():
        if goal_domain_and_others in valid_domain_list:
            # print(goal_domain_and_others)
            if (goal.get(goal_domain_and_others)):
                temp_domain_list.append(goal_domain_and_others)
                valid_fname_domain_pair_dict.update(domains=temp_domain_list)

    ### log operation
    for idx, turn in enumerate(log):
        # We only look at even turns (metadata present)
        # remove dialog_act and span_info
        if idx % 2 != 0:
            turn.pop("dialog_act", None)
            turn.pop("span_info", None)
            metadata = turn.get("metadata", {})
            # print(metadata)
            
            # Check each domain in metadata
            for domain in valid_domain_list:
                if domain not in valid_fname_domain_pair_dict['domains']:
                    if domain in metadata:
                        metadata[domain] = {
                            "book": {"booked": []}, 
                            "semi": {}
                        }
        else:
            # remove dialog_act and span_info
            turn.pop("dialog_act", None)
            turn.pop("span_info", None)


with open(output_data_path, 'w') as out_file:
    json.dump(data, out_file, indent=4)



'''
"SNG01856.json": {
        "goal": {
            "taxi": {},
            "police": {},
            "hospital": {},
            "hotel": {
                "info": {
                    "type": "hotel",
                    "parking": "yes",
                    "pricerange": "cheap",
                    "internet": "yes"
                },
                "fail_info": {},
                "book": {
                    "pre_invalid": true,
                    "stay": "2",
                    "day": "tuesday",
                    "invalid": false,
                    "people": "6"
                },
                "fail_book": {
                    "stay": "3"
                }
            },
            "topic": {
                "taxi": false,
                "police": false,
                "restaurant": false,
                "hospital": false,
                "hotel": false,
                "general": false,
                "attraction": false,
                "train": false,
                "booking": false
            },
            "attraction": {},
            "train": {},
            "message": [
                "You are looking for a <span class='emphasis'>place to stay</span>. The hotel should be in the <span class='emphasis'>cheap</span> price range and should be in the type of <span class='emphasis'>hotel</span>",
                "The hotel should <span class='emphasis'>include free parking</span> and should <span class='emphasis'>include free wifi</span>",
                "Once you find the <span class='emphasis'>hotel</span> you want to book it for <span class='emphasis'>6 people</span> and <span class='emphasis'>3 nights</span> starting from <span class='emphasis'>tuesday</span>",
                "If the booking fails how about <span class='emphasis'>2 nights</span>",
                "Make sure you get the <span class='emphasis'>reference number</span>"
            ],
            "restaurant": {}
        },
        "log": [
            {
                "text": "am looking for a place to to stay that has cheap price range it should be in a type of hotel",
                "metadata": {},
                "dialog_act": {
                    "Hotel-Inform": [
                        [
                            "Type",
                            "hotel"
                        ],
                        [
                            "Price",
                            "cheap"
                        ]
                    ]
                },
                "span_info": [
                    [
                        "Hotel-Inform",
                        "Type",
                        "hotel",
                        20,
                        20
                    ],
                    [
                        "Hotel-Inform",
                        "Price",
                        "cheap",
                        10,
                        10
                    ]
                ]
            },
            {
                "text": "Okay, do you have a specific area you want to stay in?",
                "metadata": {
                    "taxi": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "departure": "",
                            "arriveBy": ""
                        }
                    },
                    "police": {
                        "book": {
                            "booked": []
                        },
                        "semi": {}
                    },
                    "restaurant": {
                        "book": {
                            "booked": [],
                            "time": "",
                            "day": "",
                            "people": ""
                        },
                        "semi": {
                            "food": "",
                            "pricerange": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "hospital": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "department": ""
                        }
                    },
                    "hotel": {
                        "book": {
                            "booked": [],
                            "stay": "",
                            "day": "",
                            "people": ""
                        },
                        "semi": {
                            "name": "not mentioned",
                            "area": "not mentioned",
                            "parking": "not mentioned",
                            "pricerange": "cheap",
                            "stars": "not mentioned",
                            "internet": "not mentioned",
                            "type": "hotel"
                        }
                    },
                    "attraction": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "type": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "train": {
                        "book": {
                            "booked": [],
                            "people": ""
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "day": "",
                            "arriveBy": "",
                            "departure": ""
                        }
                    }
                },
                "dialog_act": {
                    "Hotel-Request": [
                        [
                            "Area",
                            "?"
                        ]
                    ]
                },
                "span_info": []
            },
            {
                "text": "no, i just need to make sure it's cheap. oh, and i need parking",
                "metadata": {},
                "dialog_act": {
                    "Hotel-Inform": [
                        [
                            "Parking",
                            "yes"
                        ]
                    ]
                },
                "span_info": []
            },
            {
                "text": "I found 1 cheap hotel for you that includes parking. Do you like me to book it?",
                "metadata": {
                    "taxi": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "departure": "",
                            "arriveBy": ""
                        }
                    },
                    "police": {
                        "book": {
                            "booked": []
                        },
                        "semi": {}
                    },
                    "restaurant": {
                        "book": {
                            "booked": [],
                            "time": "",
                            "day": "",
                            "people": ""
                        },
                        "semi": {
                            "food": "",
                            "pricerange": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "hospital": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "department": ""
                        }
                    },
                    "hotel": {
                        "book": {
                            "booked": [],
                            "stay": "",
                            "day": "",
                            "people": ""
                        },
                        "semi": {
                            "name": "not mentioned",
                            "area": "not mentioned",
                            "parking": "yes",
                            "pricerange": "cheap",
                            "stars": "not mentioned",
                            "internet": "not mentioned",
                            "type": "hotel"
                        }
                    },
                    "attraction": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "type": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "train": {
                        "book": {
                            "booked": [],
                            "people": ""
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "day": "",
                            "arriveBy": "",
                            "departure": ""
                        }
                    }
                },
                "dialog_act": {
                    "Booking-Inform": [
                        [
                            "none",
                            "none"
                        ]
                    ],
                    "Hotel-Inform": [
                        [
                            "Price",
                            "cheap"
                        ],
                        [
                            "Choice",
                            "1"
                        ],
                        [
                            "Parking",
                            "none"
                        ]
                    ]
                },
                "span_info": [
                    [
                        "Hotel-Inform",
                        "Price",
                        "cheap",
                        3,
                        3
                    ],
                    [
                        "Hotel-Inform",
                        "Choice",
                        "1",
                        2,
                        2
                    ]
                ]
            },
            {
                "text": "Yes, please. 6 people 3 nights starting on tuesday.",
                "metadata": {},
                "dialog_act": {
                    "Hotel-Inform": [
                        [
                            "Stay",
                            "3"
                        ],
                        [
                            "Day",
                            "tuesday"
                        ],
                        [
                            "People",
                            "6"
                        ]
                    ]
                },
                "span_info": [
                    [
                        "Hotel-Inform",
                        "Stay",
                        "3",
                        6,
                        6
                    ],
                    [
                        "Hotel-Inform",
                        "Day",
                        "tuesday",
                        10,
                        10
                    ],
                    [
                        "Hotel-Inform",
                        "People",
                        "6",
                        4,
                        4
                    ]
                ]
            },
            {
                "text": "I am sorry but I wasn't able to book that for you for Tuesday. Is there another day you would like to stay or perhaps a shorter stay?",
                "metadata": {
                    "taxi": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "departure": "",
                            "arriveBy": ""
                        }
                    },
                    "police": {
                        "book": {
                            "booked": []
                        },
                        "semi": {}
                    },
                    "restaurant": {
                        "book": {
                            "booked": [],
                            "time": "",
                            "day": "",
                            "people": ""
                        },
                        "semi": {
                            "food": "",
                            "pricerange": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "hospital": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "department": ""
                        }
                    },
                    "hotel": {
                        "book": {
                            "booked": [],
                            "stay": "3",
                            "day": "tuesday",
                            "people": "6"
                        },
                        "semi": {
                            "name": "not mentioned",
                            "area": "not mentioned",
                            "parking": "yes",
                            "pricerange": "cheap",
                            "stars": "not mentioned",
                            "internet": "not mentioned",
                            "type": "hotel"
                        }
                    },
                    "attraction": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "type": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "train": {
                        "book": {
                            "booked": [],
                            "people": ""
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "day": "",
                            "arriveBy": "",
                            "departure": ""
                        }
                    }
                },
                "dialog_act": {
                    "Booking-NoBook": [
                        [
                            "Day",
                            "Tuesday"
                        ]
                    ],
                    "Booking-Request": [
                        [
                            "Stay",
                            "?"
                        ],
                        [
                            "Day",
                            "?"
                        ]
                    ]
                },
                "span_info": [
                    [
                        "Booking-NoBook",
                        "Day",
                        "Tuesday",
                        14,
                        14
                    ]
                ]
            },
            {
                "text": "how about only 2 nights.",
                "metadata": {},
                "dialog_act": {
                    "Hotel-Inform": [
                        [
                            "Stay",
                            "2"
                        ]
                    ]
                },
                "span_info": [
                    [
                        "Hotel-Inform",
                        "Stay",
                        "2",
                        3,
                        3
                    ]
                ]
            },
            {
                "text": "Booking was successful.\nReference number is : 7GAWK763. Anything else I can do for you?",
                "metadata": {
                    "taxi": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "departure": "",
                            "arriveBy": ""
                        }
                    },
                    "police": {
                        "book": {
                            "booked": []
                        },
                        "semi": {}
                    },
                    "restaurant": {
                        "book": {
                            "booked": [],
                            "time": "",
                            "day": "",
                            "people": ""
                        },
                        "semi": {
                            "food": "",
                            "pricerange": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "hospital": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "department": ""
                        }
                    },
                    "hotel": {
                        "book": {
                            "booked": [
                                {
                                    "name": "the cambridge belfry",
                                    "reference": "7GAWK763"
                                }
                            ],
                            "stay": "2",
                            "day": "tuesday",
                            "people": "6"
                        },
                        "semi": {
                            "name": "not mentioned",
                            "area": "not mentioned",
                            "parking": "yes",
                            "pricerange": "cheap",
                            "stars": "not mentioned",
                            "internet": "not mentioned",
                            "type": "hotel"
                        }
                    },
                    "attraction": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "type": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "train": {
                        "book": {
                            "booked": [],
                            "people": ""
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "day": "",
                            "arriveBy": "",
                            "departure": ""
                        }
                    }
                },
                "dialog_act": {
                    "general-reqmore": [
                        [
                            "none",
                            "none"
                        ]
                    ],
                    "Booking-Book": [
                        [
                            "Ref",
                            "7GAWK763"
                        ]
                    ]
                },
                "span_info": [
                    [
                        "Booking-Book",
                        "Ref",
                        "7GAWK763",
                        8,
                        8
                    ]
                ]
            },
            {
                "text": "No, that will be all. Good bye.",
                "metadata": {},
                "dialog_act": {
                    "general-bye": [
                        [
                            "none",
                            "none"
                        ]
                    ]
                },
                "span_info": []
            },
            {
                "text": "Thank you for using our services.",
                "metadata": {
                    "taxi": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "departure": "",
                            "arriveBy": ""
                        }
                    },
                    "police": {
                        "book": {
                            "booked": []
                        },
                        "semi": {}
                    },
                    "restaurant": {
                        "book": {
                            "booked": [],
                            "time": "",
                            "day": "",
                            "people": ""
                        },
                        "semi": {
                            "food": "",
                            "pricerange": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "hospital": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "department": ""
                        }
                    },
                    "hotel": {
                        "book": {
                            "booked": [
                                {
                                    "name": "the cambridge belfry",
                                    "reference": "7GAWK763"
                                }
                            ],
                            "stay": "2",
                            "day": "tuesday",
                            "people": "6"
                        },
                        "semi": {
                            "name": "not mentioned",
                            "area": "not mentioned",
                            "parking": "yes",
                            "pricerange": "cheap",
                            "stars": "not mentioned",
                            "internet": "not mentioned",
                            "type": "hotel"
                        }
                    },
                    "attraction": {
                        "book": {
                            "booked": []
                        },
                        "semi": {
                            "type": "",
                            "name": "",
                            "area": ""
                        }
                    },
                    "train": {
                        "book": {
                            "booked": [],
                            "people": ""
                        },
                        "semi": {
                            "leaveAt": "",
                            "destination": "",
                            "day": "",
                            "arriveBy": "",
                            "departure": ""
                        }
                    }
                },
                "dialog_act": {
                    "general-bye": [
                        [
                            "none",
                            "none"
                        ]
                    ]
                },
                "span_info": []
            }
        ]
    }
    
'''