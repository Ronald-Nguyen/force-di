public with sharing class di_BindingParam {

    public static Map<String, Object> parameters { get; set; }

    @InvocableMethod(label = 'Get Value')
    public static List<Response> invoke( List<Request> requests ) {
        List<Response> responses = new List<Response>();

        for ( Request req : requests ) {
            Response res = new Response();
            if ( parameters != null ) {
                processRequest(req, res);
            } else {
                System.debug( LoggingLevel.WARN, 'BindingParam.parameters is null' );
            }
            System.debug( 'paramName=' + req );
            System.debug( res );
            responses.add( res );
        }
        return responses;
    }

    private static void processRequest(Request req, Response res) {
        Object value = parameters.get( req.paramName );
        if ( value != null ) {
            mapValueToResponse(res, value);
            synchronizeResponseFields(res);
        } else {
            System.debug( LoggingLevel.WARN, 'BindingParam.parameters.get("' + req + '") is null' );
        }
    }

    private static void mapValueToResponse(Response res, Object value) {
        if ( value instanceof Decimal ) { 
            populateFromDecimal(res, (Decimal) value);
        } else if ( value instanceof Decimal[] ) {
            populateFromDecimalArray(res, (Decimal[]) value);
        } else if ( value instanceof String ) {
            populateFromString(res, (String) value);
        } else if ( value instanceof String[] ) {
            populateFromStringArray(res, (String[]) value);
        } else if ( value instanceof Boolean ) {
            populateFromBoolean(res, (Boolean) value);
        } else if ( value instanceof Boolean[] ) {
            populateFromBooleanArray(res, (Boolean[]) value);
        } else if ( value instanceof Date ) {
            populateFromDate(res, (Date) value);
        } else if ( value instanceof Date[] ) {
            populateFromDateArray(res, (Date[]) value);
        } else if ( value instanceof DateTime ) {
            populateFromDateTime(res, (DateTime) value);
        } else if ( value instanceof DateTime[] ) {
            populateFromDateTimeArray(res, (DateTime[]) value);
        }
    }

    private static void populateFromDecimal(Response res, Decimal v) {
        res.longValue = toLongValue( v );
        res.decimalValue = v;
        res.stringValue = toStringValue( v );
        res.booleanValue = toBooleanValue( v );
        res.dateValue = toDateValue( v );
        res.dateTimeValue = toDateTimeValue( v );
    }

    private static void populateFromDecimalArray(Response res, Decimal[] vs) {
        res.decimalValues = vs;
        for ( Decimal v : vs ) {
            res.longValues.add( toLongValue( v ) );
            res.stringValues.add( toStringValue( v ) );
            res.booleanValues.add( toBooleanValue( v ) );
            res.dateValues.add( toDateValue( v ) );
            res.dateTimeValues.add( toDateTimeValue( v ) );
        }
    }

    private static void populateFromString(Response res, String v) {
        res.longValue = toLongValue( v );
        res.decimalValue = toDecimalValue( v );
        res.stringValue = v;
        res.booleanValue = toBooleanValue( v );
        res.dateValue = toDateValue( v );
        res.dateTimeValue = toDateTimeValue( v );
    }

    private static void populateFromStringArray(Response res, String[] vs) {
        res.stringValues = vs;
        for ( String v : vs ) {
            res.longValues.add( toLongValue( v ) );
            res.decimalValues.add( toDecimalValue( v ) );
            res.booleanValues.add( toBooleanValue( v ) );
            res.dateValues.add( toDateValue( v ) );
            res.dateTimeValues.add( toDateTimeValue( v ) );
        }
    }

    private static void populateFromBoolean(Response res, Boolean v) {
        res.longValue = toLongValue( v );
        res.decimalValue = toDecimalValue( v );
        res.stringValue = toStringValue( v );
        res.booleanValue = v;
        res.dateValue = toDateValue( v );
        res.dateTimeValue = toDateTimeValue( v );
    }

    private static void populateFromBooleanArray(Response res, Boolean[] vs) {
        res.booleanValues = vs;
        for ( Boolean v : vs ) {
            res.longValues.add( toLongValue( v ) );
            res.decimalValues.add( toDecimalValue( v ) );
            res.stringValues.add( toStringValue( v ) );
            res.dateValues.add( toDateValue( v ) );
            res.dateTimeValues.add( toDateTimeValue( v ) );
        }
    }

    private static void populateFromDate(Response res, Date v) {
        res.longValue = toLongValue( v );
        res.decimalValue = toDecimalValue( v );
        res.stringValue = toStringValue( v );
        res.booleanValue = toBooleanValue( v );
        res.dateValue = v;
        res.dateTimeValue = toDateTimeValue( v );
    }

    private static void populateFromDateArray(Response res, Date[] vs) {
        res.dateValues = vs;
        for ( Date v : vs ) {
            res.longValues.add( toLongValue( v ) );
            res.decimalValues.add( toDecimalValue( v ) );
            res.stringValues.add( toStringValue( v ) );
            res.booleanValues.add( toBooleanValue( v ) );
            res.dateTimeValues.add( toDateTimeValue( v ) );
        }
    }

    private static void populateFromDateTime(Response res, DateTime v) {
        res.longValue = toLongValue( v );
        res.decimalValue = toDecimalValue( v );
        res.stringValue = toStringValue( v );
        res.booleanValue = toBooleanValue( v );
        res.dateValue = toDateValue( v );
        res.dateTimeValue = v;
    }

    private static void populateFromDateTimeArray(Response res, DateTime[] vs) {
        res.dateTimeValues = vs;
        for ( DateTime v : vs ) {
            res.longValues.add( toLongValue( v ) );
            res.decimalValues.add( toDecimalValue( v ) );
            res.stringValues.add( toStringValue( v ) );
            res.booleanValues.add( toBooleanValue( v ) );
            res.dateValues.add( toDateValue( v ) );
        }
    }

    private static void synchronizeResponseFields(Response res) {
        if ( res.longValue     == null && !res.longValues.isEmpty() )     { res.longValue = res.longValues[0]; }
        if ( res.decimalValue  == null && !res.decimalValues.isEmpty() )  { res.decimalValue = res.decimalValues[0]; }
        if ( res.stringValue   == null && !res.stringValues.isEmpty() )   { res.stringValue = res.stringValues[0]; }
        if ( res.booleanValue  == null && !res.booleanValues.isEmpty() )  { res.booleanValue = res.booleanValues[0]; }
        if ( res.dateValue     == null && !res.dateValues.isEmpty() )     { res.dateValue = res.dateValues[0]; }
        if ( res.dateTimeValue == null && !res.dateTimeValues.isEmpty() ) { res.dateTimeValue = res.dateTimeValues[0]; }

        if ( res.longValue     != null && res.longValues.isEmpty() )     { res.longValues.add( res.longValue ); }
        if ( res.decimalValue  != null && res.decimalValues.isEmpty() )  { res.decimalValues.add( res.decimalValue ); }
        if ( res.stringValue   != null && res.stringValues.isEmpty() )   { res.stringValues.add( res.stringValue ); }
        if ( res.booleanValue  != null && res.booleanValues.isEmpty() )  { res.booleanValues.add( res.booleanValue ); }
        if ( res.dateValue     != null && res.dateValues.isEmpty() )     { res.dateValues.add( res.dateValue ); }
        if ( res.dateTimeValue != null && res.dateTimeValues.isEmpty() ) { res.dateTimeValues.add( res.dateTimeValue ); }
    }

    private static Long toLongValue( Decimal value ) {
        return ( value == null ? null : value.longValue() );
    }

    private static Long toLongValue( String value ) {
        return ( ( value == null || !value.isNumeric() ) ? null : Long.valueOf( value ) );
    }

    private static Long toLongValue( Boolean value ) {
        return ( value == null ? null : value ? 1 : 0 );
    }

    private static Long toLongValue( Date value ) {
        return ( value == null ? null : toLongValue(
            DateTime.newInstance( value.year(), value.month(), value.day(), 0, 0, 0 )
        ));
    }

    private static Long toLongValue( DateTime value ) {
        return ( value == null ? null : value.getTime() );
    }

    private static Decimal toDecimalValue( String value ) {
        return ( ( value == null || !value.isNumeric() ) ? null : Decimal.valueOf( value ) );
    }

    private static Decimal toDecimalValue( Boolean value ) {
        return ( value == null ? null : value ? 1 : 0 );
    }

    private static Decimal toDecimalValue( Date value ) {
        return ( value == null ? null : toDecimalValue(
            DateTime.newInstance( value.year(), value.month(), value.day(), 0, 0, 0 )
        ));
    }

    private static Decimal toDecimalValue( DateTime value ) {
        return ( value == null ? null : value.getTime() );
    }

    private static String toStringValue( Date value ) {
        return ( value == null ? null : String.valueOf( value ) );
    }

    private static String toStringValue( DateTime value ) {
        return ( value == null ? null : String.valueOf( value ) );
    }

    private static String toStringValue( Object value ) {
        return ( value == null ? null : String.valueOf( value ) );
    }

    private static Boolean toBooleanValue( Decimal value ) {
        return ( value == null ? false : ( value > 0 ) );
    }

    private static Boolean toBooleanValue( String value ) {
        return ( value == null ? false : Boolean.valueOf( value ) );
    }

    private static Boolean toBooleanValue( Date value ) {
        return ( value == null ? false : true );
    }

    private static Boolean toBooleanValue( DateTime value ) {
        return ( value == null ? false : true );
    }

    private static Date toDateValue( Decimal value ) {
        return ( value == null ? null : DateTime.newInstance( value.longValue() ).date() );
    }

    private static Date toDateValue( String value ) {
        try {
            return ( value == null ? null : Date.valueOf( value ) ); 
        } catch ( Exception e ) {
            return null;
        }
    }

    private static Date toDateValue( Boolean value ) {
        return null;
    }

    private static Date toDateValue( DateTime value ) {
        return ( value == null ? null : value.date() );
    }

    private static DateTime toDateTimeValue( Decimal value ) {
        return ( value == null ? null : DateTime.newInstance( value.longValue() ) );
    }

    private static DateTime toDateTimeValue( String value ) {
        try {
            return ( value == null ? null : DateTime.valueOf( value ) ); 
        } catch ( Exception e ) {
            return toDateValue( value ); 
        }
    }

    private static DateTime toDateTimeValue( Boolean value ) {
        return null;
    }

    private static DateTime toDateTimeValue( Date value ) {
        return ( value == null ? null : DateTime.newInstance( value.year(), value.month(), value.day(), 0, 0, 0 ) );
    }

    public class Request {
        @InvocableVariable(label = 'Parameter Name' required = true)
        public String paramName;
    }

    public class Response {
        @InvocableVariable(label = 'Number' description = 'Whole number (no decimals)')
        public Long longValue;

        @InvocableVariable(label = 'Decimal' description = 'Number that may include decimal places')
        public Decimal decimalValue;

        @InvocableVariable(label = 'Text' description = 'Text value')
        public String stringValue;

        @InvocableVariable(label = 'Boolean' description = 'true/false')
        public Boolean booleanValue;

        @InvocableVariable(label = 'Date')
        public Date dateValue;

        @InvocableVariable(label = 'Date/Time')
        public DateTime dateTimeValue;

        @InvocableVariable(label = 'Number Collection' description = 'Collection of whole numbers')
        public Long[] longValues = new Long[] {};

        @InvocableVariable(label = 'Decimal Collection' description = 'Collection of numbers that may include decimal places')
        public Decimal[] decimalValues = new Decimal[] {};

        @InvocableVariable(label = 'Text Collection' description = 'Collection of text values')
        public String[] stringValues = new String[] {};

        @InvocableVariable(label = 'Boolean Collection' description = 'Collection of true/false values')
        public Boolean[] booleanValues = new Boolean[] {};

        @InvocableVariable(label = 'Date Collection' description = 'Collection of date values')
        public Date[] dateValues = new Date[] {};

        @InvocableVariable(label = 'Date/Time Collection' description = 'Collection of date/time values')
        public DateTime[] dateTimeValues = new DateTime[] {};
    }
}