@isTest
private with sharing class di_FlowTest {
    @isTest
    public static void givenNullInputWhenObjectCreatedThenSetInitialized(){
        //Given
        Flow.Interview flow = null;
        //When
        di_Flow diObj = new di_Flow(flow);
        //Then
        System.assertEquals(0, diObj.outputVarsNames.size(), 'Set is not initialized');
        System.assertEquals(0, diObj.outputVarsNamesRequired.size(), 'Set is not initialized');
    }

    @isTest
    public static void givenStringValueWhenOutputThenStringAddedToSet(){
        //Given
        Flow.Interview flow = null;
        String variableName = 'Test';
        //When
        di_Flow diObj = new di_Flow(flow);
        diObj.output(variableName);
        //Then
        System.assertEquals(1, diObj.outputVarsNames.size(), 'value not inserted in Set');
    }

    @isTest
    public static void givenVariableNameWhenRequiredThenValueAddedToSet(){
        //Given
        Flow.Interview flow = null;
        String variableName = 'Test';
        //When
        di_Flow diObj = new di_Flow(flow);
        diObj.required(variableName);
        //Then
        System.assertEquals(1, diObj.outputVarsNames.size(), 'value not inserted in Set');
        System.assertEquals(1, diObj.outputVarsNamesRequired.size(), 'value not inserted in Set');
    }

    /** 
     * Removed until further mocking support added to di_Flow 
     *  (see https://github.com/apex-enterprise-patterns/flowextensions for ideas)
    @isTest
    public static void givenFlowNameWhenReturningThenExceptionThrown(){
        //Given
        String flowName='HelloWorld';
        Flow.Interview flow = Flow.Interview.createInterview(flowName, new Map<String, Object>());
        String variableName = 'HelloWorld';
        String outputVariableName = 'HelloWorld';
        //When
        di_Flow diObj = new di_Flow(flow);
        diObj.required(variableName);
        try{
        diObj.returning(outputVariableName);
        } catch(Exception e){
            //Then
            System.assertEquals(e.getMessage().contains('expected but not returned from Flow'), true, 'correct Exception in not thrown');
        }
    }
    **/

    @isTest
    public static void givenMultipleValuesWhenOutputThenAllAddedToSet(){
        //Given
        Flow.Interview flow = null;
        String var1 = 'Test1';
        String var2 = 'Test2';
        //When
        di_Flow diObj = new di_Flow(flow);
        diObj.output(var1);
        diObj.output(var2);
        //Then
        System.assertEquals(2, diObj.outputVarsNames.size(), 'values not inserted in Set');
    }

    @isTest
    public static void givenMultipleRequiredWhenRequiredThenAllAddedToSet(){
        //Given
        Flow.Interview flow = null;
        String var1 = 'Test1';
        String var2 = 'Test2';
        //When
        di_Flow diObj = new di_Flow(flow);
        diObj.required(var1);
        diObj.required(var2);
        //Then
        System.assertEquals(2, diObj.outputVarsNames.size(), 'values not inserted in Set');
        System.assertEquals(2, diObj.outputVarsNamesRequired.size(), 'values not inserted in Set');
    }

    @isTest
    public static void givenFlowWhenOutputCalledMultipleTimesWithSameValueThenOnlyOneAdded(){
        //Given
        Flow.Interview flow = null;
        String variableName = 'Test';
        //When
        di_Flow diObj = new di_Flow(flow);
        diObj.output(variableName);
        diObj.output(variableName);
        //Then
        System.assertEquals(1, diObj.outputVarsNames.size(), 'duplicate value should not be added');
    }

    @isTest
    public static void givenFlowWhenRequiredCalledMultipleTimesWithSameValueThenOnlyOneAdded(){
        //Given
        Flow.Interview flow = null;
        String variableName = 'Test';
        //When
        di_Flow diObj = new di_Flow(flow);
        diObj.required(variableName);
        diObj.required(variableName);
        //Then
        System.assertEquals(1, diObj.outputVarsNames.size(), 'duplicate value should not be added');
        System.assertEquals(1, diObj.outputVarsNamesRequired.size(), 'duplicate value should not be added');
    }

    @isTest
    public static void givenFlowWhenOutputAndRequiredCalledWithSameValueThenBothSetsContainValue(){
        //Given
        Flow.Interview flow = null;
        String variableName = 'Test';
        //When
        di_Flow diObj = new di_Flow(flow);
        diObj.output(variableName);
        diObj.required(variableName);
        //Then
        System.assertEquals(1, diObj.outputVarsNames.size(), 'value should be in output set');
        System.assertEquals(1, diObj.outputVarsNamesRequired.size(), 'value should be in required set');
    }

    @isTest
    public static void givenFlowWhenEmptyStringOutputThenValueAddedToSet(){
        //Given
        Flow.Interview flow = null;
        String variableName = '';
        //When
        di_Flow diObj = new di_Flow(flow);
        diObj.output(variableName);
        //Then
        System.assertEquals(1, diObj.outputVarsNames.size(), 'empty string should be added');
    }
}
